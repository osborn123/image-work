"""
定义没有reference情况下的方案
首先采用KMEANS的方式来分别聚类得到k个中心坐标（可以用手肘法尝试不同的k）。
其次，针对k个节点相互的距离来计算得到一个k-clique的图，再通过permutation的方式来获得不同的距离之差？
"""
import itertools
import argparse
import tqdm
import numpy as np
import faiss, time, random
import torch, math
import logging, os
import wandb
# from pytorch_lightning import seed_everything
from utils import cal_recall, select_random_features
from transform.linear_transform import LinearTransform
from transform.MMD_transform import MMDTransform
from transform.kmeans_reduction import KMeansReduction
from torchvision import datasets, transforms, models
from generate_feature.test_lp import train_model, load_data
from sklearn.cluster import KMeans

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
    return feature1, feature2, dist

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cal_diff(k_cluster_1_dist, k_cluster_2_dist, mapping):
    k_cluster_1_dist = k_cluster_1_dist[np.ix_(mapping, mapping)]
    k_cluster_diff = np.abs(k_cluster_1_dist - k_cluster_2_dist)
    return k_cluster_diff.mean()

def main(args):
    np.random.seed(0)
    if args.run_entity != "":
        wandb.init(project="multi_source_retrieval", entity=args.run_entity, config=args.__dict__, mode=args.mode)
    else:
        wandb.init(project="multi_source_retrieval", config=args.__dict__, mode="disabled")
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading features...")
    # feature1, feature2, feature3 = load_three_features(args)
    feature1, feature2, dist = load_two_features_and_dist(args)

    # feature1.shape [2708, 128]; feature2.shape [2708, 128]
    logging.info(f"feature1.shape {feature1.shape}; feature2.shape {feature2.shape}")

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
    logging.info(f"diff_result: {diff_result}")

    return [], ""

if __name__ == "__main__":
    seed_everything(42)
    # 用parser引导参数
    parser = argparse.ArgumentParser()
    # parser.add_argument("--reduced_dim", type=int, default=128, help="0=>no reduction, other=> call kmeans to reduction")
    parser.add_argument("--reduced_dim", type=int, default=0, help="0=>no reduction, other=> call kmeans to reduction")

    parser.add_argument("--run_entity", type=str, default="", help="run entity")
    parser.add_argument("--mode", type=str, default="disabled", help="enabled or disabled")

    parser.add_argument("--saved_features_dir", type=str, default="saved_features", help="saved features dir")
    # parser.add_argument("--feature1_path", type=str, default="inception_mnist_features.pt", help="feature1 path")
    # parser.add_argument("--feature2_path", type=str, default="vgg_mnist_reduced_features.pt", help="feature2 path")
    # parser.add_argument("--feature3_path", type=str, default="inception_mnist_features.pt", help="feature3 path")
    parser.add_argument("--gnn_model1", type=str, default="GCN", help="GCN or GAT")
    parser.add_argument("--gnn_model2", type=str, default="GAT", help="GCN or GAT")
    parser.add_argument("--loss_type1", type=str, default="MSE", help="MSE or MAE or MAX or MIN or SUM or MEAN")
    parser.add_argument("--loss_type2", type=str, default="MSE", help="MSE or MAE or MAX or MIN or SUM or MEAN")

    parser.add_argument("--k1", type=int, default=10, help="k1 for kmeans")
    parser.add_argument("--k2", type=int, default=10, help="k2 for kmeans")

    parser.add_argument("--feature3_path", type=str, default="GAT_cora.pt", help="feature3 path")

    parser.add_argument("--mapping_method", type=str, default="random", help="")
    # parser.add_argument("--mapping_method", type=str, default="permutation", help="")

    parser.add_argument("--dist_path", type=str, default="shortestpath/native_dis.pt", help="sp groundtruth dist")

    parser.add_argument("--select_method", type=str, default="random", help="random") # random full
    # parser.add_argument("--select_proportion", type=float, default=0.5, help="select proportion")
    # parser.add_argument("--select_proportion", type=float, default=1.0, help="select proportion")
    # parser.add_argument("--select_proportion", type=float, default=0.2, help="select proportion")
    parser.add_argument("--select_proportion", type=float, default=0.9, help="select proportion")
    # parser.add_argument("--select_proportion", type=float, default=0.0, help="select proportion")

    parser.add_argument("--transform_method", type=str, default="linear", help="linear")
    parser.add_argument("--transform_epoch", type=int, default=10000, help="epoch for transform (only for linear now)")
    parser.add_argument("--transform_lr", type=float, default=1e-6, help="learning rate for transform (only for linear now)")
    parser.add_argument("--transform_batch_size", type=int, default=256, help="batch size for transform (only for linear now)")

    parser.add_argument("--candidate_num", type=int, default=10, help="the candidate_num is the m in the m@recall")
    parser.add_argument("--search_range", type=int, default=[10, 20, 50, 100, 500, 1000], help="the search range for faiss search")
    # overlap_ratio
    parser.add_argument("--overlap_ratio", type=float, default=0.3, help="overlap ratio")

    parser.add_argument("--device", type=int, default=0, help="device for training")
    args = parser.parse_args()
    args.feature1_path = f"shortestpath_cross/shortest_path_{args.gnn_model1}Encoder_{args.loss_type1}_me.pt"
    args.feature2_path = f"shortestpath_cross/shortest_path_{args.gnn_model2}Encoder_{args.loss_type2}_me.pt"
    print(args)
    ans = main(args)
    print(ans)
