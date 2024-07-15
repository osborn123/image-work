import argparse
import numpy as np
import faiss, time, random
import torch
import logging, os
import wandb
# from pytorch_lightning import seed_everything
from utils import cal_recall, select_random_features
from transform.linear_transform import LinearTransform
from transform.MMD_transform import MMDTransform
from transform.kmeans_reduction import KMeansReduction
from torchvision import datasets, transforms, models
from generate_feature.test_lp import train_model, load_data
import math
import itertools
import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans


def cal_diff(k_cluster_1_dist, k_cluster_2_dist, mapping):
    k_cluster_1_dist = k_cluster_1_dist[np.ix_(mapping, mapping)]
    k_cluster_diff = np.abs(k_cluster_1_dist - k_cluster_2_dist)
    return k_cluster_diff.mean()

def get_guided_by_no_reference(args, feature1, feature2):
    # 开始使用KMeans来聚类
    k_cluster_1 = KMeans(n_clusters=args.k1, random_state=0).fit(feature1)
    k_cluster_1_centers = k_cluster_1.cluster_centers_

    k_cluster_2 = KMeans(n_clusters=args.k2, random_state=0).fit(feature2)
    k_cluster_2_centers = k_cluster_2.cluster_centers_

    # 两组聚类中心分别计算内部的距离
    k_cluster_1_dist = np.linalg.norm(k_cluster_1_centers[:, np.newaxis] - k_cluster_1_centers, axis=2)
    k_cluster_2_dist = np.linalg.norm(k_cluster_2_centers[:, np.newaxis] - k_cluster_2_centers, axis=2)

    logging.info(f"The numeric range of k_cluster_1_dist is [{k_cluster_1_dist.min()}, {k_cluster_1_dist.max()}]")
    logging.info(f"The numeric range of k_cluster_2_dist is [{k_cluster_2_dist.min()}, {k_cluster_2_dist.max()}]")

    # 通过mapping函数来找到两组聚类中心之间的对应关系
    if args.mapping_method == "random":
        # 随机生成args.k1 => args.k2之间的映射关系, 这是多对一的关系
        mapping = np.random.randint(0, args.k2, size=args.k1)
    elif args.mapping_method == "permutation":
        min_diff = float("inf")
        total_permutations = math.factorial(args.k2) / math.factorial(args.k2 - args.k1)
        for mapping in tqdm.tqdm(itertools.permutations(range(args.k2), args.k1), total=total_permutations): # 先针对1对1的关系
            diff = cal_diff(k_cluster_1_dist, k_cluster_2_dist, mapping)
            if diff < min_diff:
                min_diff = diff
                best_mapping = mapping
        mapping = best_mapping
            
    # 通过mapping来计算两组聚类中心构成的图的pairwise的距离的差值
    diff_result = cal_diff(k_cluster_1_dist, k_cluster_2_dist, mapping)
    wandb.log({"diff_result": diff_result})
    logging.info(f"diff_result: {diff_result}")
    return k_cluster_1_centers[mapping, :], k_cluster_2_centers

def calculate_errors(dist, max_value, dist_normalized, prefix):
    # 计算AE
    ae = torch.abs(dist_normalized - dist)

    # 计算MAE
    mae = ae.mean()

    # 计算ME
    me = ae.max()

    # 计算MRE
    # mre = (ae / dist).mean() if dist.sum() != 0 else torch.tensor(0.0)
    mask = dist != 0
    mre = (torch.masked_select(ae, mask) / torch.masked_select(dist, mask)).mean()


    # 计算MSE
    mse = torch.nn.functional.mse_loss(dist_normalized, dist)

    # 计算RMSE
    rmse = torch.sqrt(mse)

    return {
        f"【AE】{prefix}": np.round(ae.mean().item(), 4), 
        f"【MAE】{prefix}": np.round(mae.item(), 4), 
        f"【ME】{prefix}": np.round(me.item(), 4), 
        f"【MRE】{prefix}": np.round(mre.item(), 4), 
        f"【MSE】{prefix}": np.round(mse.item(), 4), 
        f"【RMSE】{prefix}": np.round(rmse.item(), 4)
    }

def compute_ae_loss(z, sp_dist, normalize="max"):
    # sp_dist = data.sp_dist_raw
    if normalize == "max":
        dist = torch.cdist(z, z, p=2)
        dist = dist / dist.max()
        dist *= sp_dist.max()
    ae_loss = torch.abs(sp_dist - dist).mean()
    return ae_loss

def compute_mse_loss(z, sp_dist):
    return F.mse_loss(z/z.max(), sp_dist/sp_dist.max())

def cal_ae_loss(dist, target_dist):
    ae = torch.abs(dist - target_dist)
    return ae.mean()

def cal_me_loss(dist, target_dist):
    ae = torch.abs(dist - target_dist)
    return ae.max()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def test_lp(args, feature1, feature2, feature2_selected, only_1_index, only_2_index, overlap_index, start_time):
    model_name1, data1 = args.feature1_path.replace(".pt", "").split("_")
    model_name2, data2 = args.feature2_path.replace(".pt", "").split("_")
    if data1 == data2:
        data = load_data(data1, device="cpu")
    else:
        raise ValueError("data not match")
    
    _, model1 = train_model(encoder_name=model_name1)
    # _, model2 = train_model(encoder_name=model_name2)

    new_feature = feature1.copy()
    new_feature[overlap_index] = feature2_selected[overlap_index]
    # new_feature1 = torch.from_numpy(np.concatenate)
    auc_ori, ap_ori = model1.test(torch.from_numpy(feature1), data.test_pos_edge_index, data.test_neg_edge_index)
    auc, ap = model1.test(torch.from_numpy(new_feature), data.test_pos_edge_index, data.test_neg_edge_index)
    # return (auc_ori, ap_ori), (auc, ap)
    return {"auc_ori": auc_ori, "ap_ori": ap_ori, "auc_tranformed": auc, "ap_transformed": ap}

def test_sp(args, feature1, feature2, feature1_transformed, only_1_index, only_2_index, overlap_index, start_time, dist, normalization="fix"):
    
    feature1 = torch.from_numpy(feature1)
    feature2 = torch.from_numpy(feature2)
    feature1_transformed = torch.from_numpy(feature1_transformed)
    dist = torch.from_numpy(dist.copy())

    dist1 = torch.cdist(feature1, feature1, p=2)
    dist2 = torch.cdist(feature2, feature2, p=2)
    dist1_selected = torch.cdist(feature1_transformed, feature1_transformed, p=2)
    dist12 = torch.cdist(feature1, feature2, p=2)
    dist3 = torch.cdist(feature1_transformed, feature2, p=2)

    if normalization == "max" or normalization == "fix":
        dist1 = (dist1 / dist1.max() * dist.max())
        dist2 = (dist2 / dist2.max() * dist.max())
        dist1_selected = (dist1_selected / dist1_selected.max() * dist.max())
        dist12 = (dist12 / dist12.max() * dist.max())
        dist3 = (dist3 / dist3.max() * dist.max())
    elif normalization == "unnorma":
        max1_value = 1
        max2_value = 1
        max2_selected_value = 1
        max3 = 1
    
    return {"ae_dist1": cal_ae_loss(dist1, dist), "ae_dist2": cal_ae_loss(dist2, dist), "ae_dist2_selected": cal_ae_loss(dist1_selected, dist), 
            "ae_dist21": cal_ae_loss(dist12, dist),
            "ae_dist3": cal_ae_loss(dist3, dist),
            "mse_dist1": compute_mse_loss(dist1, dist), "mse_dist2": compute_mse_loss(dist2, dist), "mse_dist2_selected": compute_mse_loss(dist1_selected, dist),
            "mse_dist21": compute_mse_loss(dist12, dist), "mse_dist3": compute_mse_loss(dist3, dist),
            "me_dist1": cal_me_loss(dist1, dist), "me_dist2": cal_me_loss(dist2, dist), "me_dist2_selected": cal_me_loss(dist1_selected, dist),
            "me_dist21": cal_me_loss(dist12, dist), "me_dist3": cal_me_loss(dist3, dist),}


def test_reall(args, feature1, feature2, feature2_selected, only_1_index, only_2_index, start_time):
    logging.info("Creating index...")
    index_feature1 = faiss.IndexFlatL2(feature1.shape[1])
    index_feature2 = faiss.IndexFlatL2(feature2.shape[1])

    logging.info("Adding features to index...")
    # index_feature1.add(feature1[only_2_index])
    index_feature1.add(feature1[only_2_index])
    index_feature2.add(feature2_selected[only_2_index])
    # index_feature2.add(feature2[only_2_index])

    recall_list = []
    for search_range in args.search_range:
        logging.info("Performing search...")
        ans1 = index_feature1.search(feature1[:args.candidate_num], search_range)
        ans2 = index_feature2.search(feature1[:args.candidate_num], search_range)
        
        logging.info("Calculating recall...")
        recall = cal_recall(ans1, ans2)
        recall_list.append(recall)
        wandb.log({f"recall@{search_range}": recall})
        print(f"{args.candidate_num}@{search_range} recall: {recall}")

    end_time = time.time()
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
    wandb.log({"time": end_time - start_time})
    return recall_list

def split_through_random(feature_matrix1, feature_matrix2, proportion):
    """
    随机选择一定比例的特征。
    
    :param feature_matrix: numpy array, 特征矩阵，假设形状为 (n_samples, n_features)
    :param proportion: float, 要保留的特征比例，介于 0 和 1 之间
    :return: 保留的特征子集的numpy array
    """
    if proportion > 0.95:
        print("proportion is too large, using whole features.")
        return np.arange(feature_matrix1.shape[0]), np.arange(feature_matrix2.shape[0]), np.arange(feature_matrix1.shape[0])
    assert feature_matrix1.shape[0] == feature_matrix2.shape[0]
    n_samples = feature_matrix1.shape[0]
    n_select = int(n_samples * proportion)  # 计算要保留的特征数量
    
    # 生成所有特征的索引
    all_indices = np.arange(n_samples)
    # 随机选择指定数量的特征索引
    overlap_index = np.random.choice(all_indices, n_select, replace=False)
    # 剩下的平均分配到两个集合中
    # only_1_index = np.setdiff1d(all_indices, overlap_index)[:n_select // 2]
    only_1_index = np.array(list(set(all_indices) - set(overlap_index)))[:(n_samples - n_select) // 2]
    only_2_index = np.array(list(set(all_indices) - set(overlap_index) - set(only_1_index)))
    
    # 返回选中的特征子集
    return only_1_index, only_2_index, overlap_index

def split_through_class(feature1, label=None, label_classes=2, overlap_ratio:float=0.3):
    # 计算每个类别的样本数量
    class_counts = np.bincount(label)

    # 初始化索引数组
    only_1_index = []
    only_2_index = []
    overlap_index = []

    # 为每个类别分配索引
    for i, label_num in zip(np.unique(label), class_counts):
        # 计算每个类别应该抽取的索引数量
        overlap_class_size = int(overlap_ratio * label_num)
        
        # 抽取重叠索引
        overlap_indices = np.random.choice(np.where(label == i)[0], size=overlap_class_size, replace=False)
        overlap_index.extend(overlap_indices)
        
        # 剩余的索引分配给只有第一个特征或只有第二个特征
        remaining_indices = np.setdiff1d(np.where(label == i)[0], overlap_indices)
        
        # 均匀分配剩余索引到两个集合中
        split_size = len(remaining_indices) // 2
        only_1_index.extend(remaining_indices[:split_size])
        only_2_index.extend(remaining_indices[split_size:])

    # 将索引数组转换为 NumPy 数组
    only_1_index = np.array(only_1_index)
    only_2_index = np.array(only_2_index)
    overlap_index = np.array(overlap_index)
    return only_1_index, only_2_index, overlap_index

def load_two_features_and_dist(args, norm=False):
    if args.k != -1:
        args.k1 = args.k
        args.k2 = args.k
    args.feature1_path = f"shortestpath_cross/shortest_path_{args.gnn_model1}Encoder_{args.loss_type1}_me.pt"
    args.feature2_path = f"shortestpath_cross/shortest_path_{args.gnn_model2}Encoder_{args.loss_type2}_me.pt"
    # load feature
    if args.feature1_path.endswith(".npy"):
        feature1 = np.load(os.path.join(args.saved_features_dir, args.feature1_path))
    else:
        feature1 = torch.load(os.path.join(args.saved_features_dir, args.feature1_path), map_location="cpu")
        feature1 = feature1.numpy()
    
    if args.feature2_path.endswith(".npy"):
        feature2 = np.load(os.path.join(args.saved_features_dir, args.feature2_path))
    else:
        feature2 = torch.load(os.path.join(args.saved_features_dir, args.feature2_path), map_location="cpu")
        feature2 = feature2.numpy()

    if args.feature3_path.endswith(".npy"):
        dist = np.load(os.path.join(args.saved_features_dir, args.dist_path))
    else:
        dist = torch.load(os.path.join(args.saved_features_dir, args.dist_path), map_location="cpu")
        dist = dist.numpy()
    
    feature1 = feature1.reshape(feature1.shape[0], -1)
    feature2 = feature2.reshape(feature2.shape[0], -1)
    dist = dist.reshape(dist.shape[0], -1)
    if norm:
        feature1 = feature1 / np.linalg.norm(feature1, axis=1, keepdims=True)
        feature2 = feature2 / np.linalg.norm(feature2, axis=1, keepdims=True)
        # feature3 = feature3 / np.linalg.norm(feature3, axis=1, keepdims=True)
    print(f"【feature 1】: {compute_ae_loss(torch.from_numpy(feature1), torch.from_numpy(dist))}")
    print(f"【feature 2】: {compute_ae_loss(torch.from_numpy(feature2), torch.from_numpy(dist))}")
    return feature1, feature2, dist

def calculate_mapping(args, feature1, feature2, only_1_index, only_2_index, overlap_index):
    """
    Perform linear transform on feature2 to align it with feature1. 
    """
    if args.transform_method == "linear":
        logging.info("Performing linear transform...")
        feature1_overlapd = feature1[overlap_index] 
        feature2_overlapd = feature2[overlap_index]
        # feature2_selected[only_2_index] = LinearTransform(args, feature2_overlapd, feature1_overlapd, transformed_feature=feature2_selected[only_2_index], test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
        if len(overlap_index) > 0:
            # feature2_transformed = LinearTransform(args, feature2_overlapd, feature1_overlapd, transformed_feature=feature2_transformed, test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
            feature1_transformed = LinearTransform(args, origin=feature1_overlapd, target=feature2_overlapd, transformed_candidate=feature1, test_origin=feature1[only_1_index], test_target=feature2[only_1_index])
        else:
            feature1_transformed = feature1.copy()
    elif args.transform_method == "class_linear":
        feature2_selected = feature2.copy()
    elif args.transform_method == "" or args.transform_method == "none":
        feature2_selected = feature2
    elif args.transform_method == "HNSW":
        feature2_selected = feature2.copy()
        feature1_guided, feature2_guided = get_guided_by_no_reference(args, feature1, feature2)
        # feature2_transformed = LinearTransform(args, feature2_guided, feature1_guided, transformed_feature=feature2, test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
        feature1_transformed = LinearTransform(args, origin=feature1_guided, target=feature2_guided, transformed_candidate=feature1, test_origin=feature1[only_1_index], test_target=feature2[only_1_index])
    elif args.transform_method == "MMD":
        logging.info("Performing MMD transform...")
        # feature2_selected = feature2.copy()
        feature1_selected = feature1.copy()
        feature1_overlapd = feature1[overlap_index]
        np.random.shuffle(overlap_index)
        feature2_overlapd = feature2[overlap_index]
        # feature1_transformed, _ = MMDTransform(args, feature1_overlapd, feature2_overlapd, transformed_feature1=feature1, transformed_feature2=feature2, test_origin=feature2[only_1_index], test_target=feature1[only_1_index])
        if len(overlap_index) > 0:
            feature1_transformed = MMDTransform(args, origin=feature1_overlapd, target=feature2_overlapd, transformed_candidate=feature1, test_origin=feature1[only_1_index], test_target=feature2[only_1_index])
        else:
            feature1_transformed = feature1.copy()

    else:
        raise ValueError("transform method not found")
    return feature1_transformed

def load_two_features(args):
    # load feature
    args.saved_features_dir += "/link_prediction"
    if args.feature1_path.endswith(".npy"):
        feature1 = np.load(os.path.join(args.saved_features_dir, args.feature1_path))
    else:
        feature1 = torch.load(os.path.join(args.saved_features_dir, args.feature1_path), map_location="cpu")
        feature1 = feature1.numpy()
    
    if args.feature2_path.endswith(".npy"):
        feature2 = np.load(os.path.join(args.saved_features_dir, args.feature2_path))
    else:
        feature2 = torch.load(os.path.join(args.saved_features_dir, args.feature2_path), map_location="cpu")
        feature2 = feature2.numpy()

    feature1 = feature1.reshape(feature1.shape[0], -1)
    feature2 = feature2.reshape(feature2.shape[0], -1)
    return feature1, feature2

def main(args):
    np.random.seed(0)
    if args.run_entity != "":
        wandb.init(project="multi_source_retrieval", entity=args.run_entity, config=args.__dict__, mode="online")
    else:
        wandb.init(project="multi_source_retrieval", config=args.__dict__, mode="disabled")
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading features...")

    # load features
    if args.task == "shortest_path":
        feature1, feature2, dist = load_two_features_and_dist(args)
    elif args.task == "link_prediction":
        feature1, feature2  = load_two_features(args)

    # reduce dimension if needed
    if args.reduced_dim > 0:
        logging.info("Performing KMeans reduction...")
        feature1, feature2, feature3 = KMeansReduction(args, [feature1, feature2, feature3])

    # assign the overlap
    only_1_index, only_2_index, overlap_index = split_through_random(feature1, feature2, proportion=args.select_proportion)
    wandb.log({"overlap_index": len(overlap_index), "only_1_index": len(only_1_index), "only_2_index": len(only_2_index)})

    # do the transform (mapping)
    feature1_transformed = calculate_mapping(args, feature1, feature2, only_1_index, only_2_index, overlap_index)

    from pprint import pprint
    sp_ans = test_sp(args, feature1, feature2, feature1_transformed, only_1_index, only_2_index, overlap_index, start_time, dist, normalization="max")
    wandb.log({"max_"+k: v for k, v in sp_ans.items()})
    print(f"Max_distance: overlap_ratio: {args.select_proportion} overlap_number: {len(overlap_index)}/{feature1.shape[0]}")
    pprint(sp_ans)

if __name__ == "__main__":
    seed_everything(42)
    # 用parser引导参数
    parser = argparse.ArgumentParser()
    # parser.add_argument("--reduced_dim", type=int, default=128, help="0=>no reduction, other=> call kmeans to reduction")
    parser.add_argument("--reduced_dim", type=int, default=0, help="0=>no reduction, other=> call kmeans to reduction")

    parser.add_argument("--task", type=str, default="link_prediction", choices=['shortest_distance', 'link_prediction', 'cv', 'nlp'], help="")
    parser.add_argument("--run_entity", type=str, default="", help="run entity")
    parser.add_argument("--mode", type=str, default="disabled", help="enabled or disabled")

    parser.add_argument("--saved_features_dir", type=str, default="saved_features", help="saved features dir")
    # parser.add_argument("--feature1_path", type=str, default="inception_mnist_features.pt", help="feature1 path")
    # parser.add_argument("--feature2_path", type=str, default="vgg_mnist_reduced_features.pt", help="feature2 path")
    # parser.add_argument("--feature3_path", type=str, default="inception_mnist_features.pt", help="feature3 path")
    parser.add_argument("--gnn_model1", type=str, default="GAT", help="GCN or GAT")
    parser.add_argument("--gnn_model2", type=str, default="GraphSAGE", help="GCN or GAT")
    parser.add_argument("--loss_type1", type=str, default="MSE", help="MSE or MAE or MAX or MIN or SUM or MEAN")
    parser.add_argument("--loss_type2", type=str, default="MSE", help="MSE or MAE or MAX or MIN or SUM or MEAN")
    # parser.add_argument("--loss_type2", type=str, default="MAX", help="MSE or MAE or MAX or MIN or SUM or MEAN")

    parser.add_argument("--k1", type=int, default=10, help="k1 for kmeans")
    parser.add_argument("--k2", type=int, default=10, help="k2 for kmeans")
    parser.add_argument("--k", type=int, default=-1, help="k2 for kmeans")

    # parser.add_argument("--feature1_path", type=str, default="shortestpath/shortest_path_GATEncoder_ae_10.4974.pt", help="feature3 path")
    # parser.add_argument("--feature2_path", type=str, default="shortestpath/shortest_path_GCNEncoder_ae_11.0291.pt", help="feature3 path")
    # parser.add_argument("--feature1_path", type=str, default="shortestpath/shortest_path_GATEncoder_ae_10.4974.pt", help="feature3 path")
    # parser.add_argument("--feature2_path", type=str, default="shortestpath/shortest_path_GCNEncoder_ae_11.0291.pt", help="feature3 path")
    # parser.add_argument("--feature1_path", type=str, default="shortestpath/shortest_path_GATEncoder_no_norm.pt", help="feature3 path")
    # parser.add_argument("--feature2_path", type=str, default="shortestpath/shortest_path_GCNEncoder_no_norm.pt", help="feature3 path")
    parser.add_argument("--feature1_path", type=str, default="GAT_cora.pt", help="feature1 path")
    parser.add_argument("--feature2_path", type=str, default="GCN_cora.pt", help="feature2 path")
    parser.add_argument("--feature3_path", type=str, default="GAT_cora.pt", help="feature3 path")

    parser.add_argument("--dist_path", type=str, default="shortestpath/native_dis.pt", help="sp groundtruth dist")

    parser.add_argument("--select_method", type=str, default="random", help="random") # random full
    parser.add_argument("--select_proportion", type=float, default=0.9, help="select proportion")
    # parser.add_argument("--select_proportion", type=float, default=0.0, help="select proportion")

    parser.add_argument("--transform_method", type=str, default="linear", help="linear")
    # parser.add_argument("--transform_method", type=str, default="HNSW", help="linear")
    # parser.add_argument("--transform_method", type=str, default="MMD", help="linear")
    # parser.add_argument("--transform_method", type=str, default="class_linear", help="linear")
    # parser.add_argument("--transform_method", type=str, default="none", help="linear")
    # parser.add_argument("--transform_epoch", type=int, default=10000, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_epoch", type=int, default=100, help="epoch for transform (only for linear now)")
    parser.add_argument("--mapping_method", type=str, default="permutation", help="random or permutation")
    # parser.add_argument("--mapping_method", type=str, default="random", help="random or permutation")
    parser.add_argument("--transform_epoch", type=int, default=1000, help="epoch for transform (only for linear now)")
    parser.add_argument("--transform_lr", type=float, default=0.0001, help="learning rate for transform (only for linear now)")
    # parser.add_argument("--transform_lr", type=float, default=1e-6, help="learning rate for transform (only for linear now)")
    parser.add_argument("--transform_batch_size", type=int, default=256, help="batch size for transform (only for linear now)")

    parser.add_argument("--with_bias", type=int, default=1, help="with bias for linear transform")
    parser.add_argument("--layers", type=int, default=2, help="device for training")
    parser.add_argument("--with_activation", type=int, default=0, help="device for training")

    parser.add_argument("--candidate_num", type=int, default=10, help="the candidate_num is the m in the m@recall")
    parser.add_argument("--search_range", type=int, default=[10, 20, 50, 100, 500, 1000], help="the search range for faiss search")
    # overlap_ratio
    parser.add_argument("--overlap_ratio", type=float, default=0.3, help="overlap ratio")

    parser.add_argument("--device", type=int, default=0, help="device for training")
    args = parser.parse_args()
    print(args)
    ans = main(args)
    print(ans)