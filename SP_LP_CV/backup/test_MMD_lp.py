import argparse
import numpy as np
import faiss, time
import torch
import logging, os
import wandb
from pytorch_lightning import seed_everything
from utils import cal_recall, select_random_features
from transform.linear_transform import LinearTransform
from transform.MMD_transform import MMDTransform
from transform.kmeans_reduction import KMeansReduction
from torchvision import datasets, transforms, models
from generate_feature.test_lp import train_model, load_data

# # 定义数据预处理
# transform = transforms.Compose([
#     transforms.Resize(299),  # 调整图像大小以适应InceptionV3的输入
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将灰度图像转换为彩色图像
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
# ])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# label = train_dataset.targets.numpy()


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
    assert feature_matrix1.shape[0] == feature_matrix2.shape[0]
    n_samples = feature_matrix1.shape[0]
    n_select = int(n_samples * proportion)  # 计算要保留的特征数量
    
    # 生成所有特征的索引
    all_indices = np.arange(n_samples)
    # 随机选择指定数量的特征索引
    overlap_index = np.random.choice(all_indices, n_select, replace=False)
    # 剩下的平均分配到两个集合中
    only_1_index = np.setdiff1d(all_indices, overlap_index)[:n_select // 2]
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

def load_three_features(args, norm=True):
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
        feature3 = np.load(os.path.join(args.saved_features_dir, args.feature3_path))
    else:
        feature3 = torch.load(os.path.join(args.saved_features_dir, args.feature3_path), map_location="cpu")
        feature3 = feature3.numpy()
    
    feature1 = feature1.reshape(feature1.shape[0], -1)
    feature2 = feature2.reshape(feature2.shape[0], -1)
    feature3 = feature3.reshape(feature3.shape[0], -1)
    if norm:
        feature1 = feature1 / np.linalg.norm(feature1, axis=1, keepdims=True)
        feature2 = feature2 / np.linalg.norm(feature2, axis=1, keepdims=True)
        feature3 = feature3 / np.linalg.norm(feature3, axis=1, keepdims=True)
    return feature1, feature2, feature3

def main(args):
    np.random.seed(0)
    if args.run_entity != "":
        wandb.init(project="multi_source_retrieval", entity=args.run_entity, config=args.__dict__, mode=args.mode)
    else:
        wandb.init(project="multi_source_retrieval", config=args.__dict__, mode="disabled")
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading features...")
    feature1, feature2, feature3 = load_three_features(args)
    
    if args.reduced_dim > 0:
        logging.info("Performing KMeans reduction...")
        feature1, feature2, feature3 = KMeansReduction(args, [feature1, feature2, feature3])

    # let two feature (1, 2) have the different objects, with some overlap
    # random version
    # only_1_index, only_2_index, overlap_index = np.split(np.random.permutation(feature1.shape[0]), [int(feature1.shape[0] * 0.3), int(feature1.shape[0] * 0.6)])
    # split index through the class label
    # only_1_index, only_2_index, overlap_index = split_through_class(feature1, label, overlap_ratio=args.overlap_ratio)
    only_1_index, only_2_index, overlap_index = split_through_random(feature1, feature2, proportion=args.select_proportion)

    feature1_selected2origin_dict = {i: j for i, j in enumerate(np.concatenate([only_1_index, overlap_index]))}
    feature2_selected2origin_dict = {i: j for i, j in enumerate(np.concatenate([only_2_index, overlap_index]))}

    if args.transform_method == "linear":
        logging.info("Performing linear transform...")
        # reset the two features, and save the index transform
        # feature1_selected =  feature1[np.concatenate([only_1_index, overlap_index])]
        # feature2_selected =  feature2[np.concatenate([only_2_index, overlap_index])]
        feature2_selected = feature2.copy()
        feature1_overlapd = feature1[overlap_index] 
        feature2_overlapd = feature2[overlap_index]
        feature2_selected[only_2_index] = LinearTransform(args, feature2_overlapd, feature1_overlapd, transformed_feature=feature2_selected[only_2_index], test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
    elif args.transform_method == "class_linear":
        # for class_index in 
        # index_2 = np.concatenate([only_2_index, overlap_index])
        feature2_selected = feature2.copy()
        for class_index in np.unique(label):
            overlap_index_per_cls = overlap_index[label[overlap_index] == class_index]
            index_2_cls = only_2_index[label[only_2_index] == class_index]
            feature2_selected[index_2_cls] = LinearTransform(args, feature2[overlap_index_per_cls], feature1[overlap_index_per_cls], transformed_feature=feature2[index_2_cls], test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
    elif args.transform_method == "" or args.transform_method == "none":
        feature2_selected = feature2
    elif args.transform_method == "MMD":
        logging.info("Performing MMD transform...")
        feature2_selected = feature2.copy()
        feature1_overlapd = feature1[overlap_index]
        np.random.shuffle(overlap_index)
        feature2_overlapd = feature2[overlap_index]
        # feature1, feature2_selected = MMDTransform(args, feature1, feature2_selected, transformed_feature1=feature1, transformed_feature2=feature2, test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
        feature1, feature2_selected = MMDTransform(args, feature2_overlapd, feature1_overlapd, transformed_feature1=feature1, transformed_feature2=feature2, test_origin=feature2[only_2_index], test_target=feature1[only_2_index])

    else:
        raise ValueError("transform method not found")
    
    recall_list = test_reall(args, feature1, feature2, feature2_selected, only_1_index, only_2_index, start_time)
    lp_ans = test_lp(args, feature1, feature2, feature2_selected, only_1_index, only_2_index, overlap_index, start_time)
    return recall_list, lp_ans


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
    parser.add_argument("--feature1_path", type=str, default="GCN_cora.pt", help="feature3 path")
    parser.add_argument("--feature2_path", type=str, default="GAT_cora.pt", help="feature3 path")
    parser.add_argument("--feature3_path", type=str, default="GAT_cora.pt", help="feature3 path")

    parser.add_argument("--select_method", type=str, default="random", help="random") # random full
    parser.add_argument("--select_proportion", type=float, default=0.5, help="select proportion")

    parser.add_argument("--transform_method", type=str, default="linear", help="linear")
    # parser.add_argument("--transform_method", type=str, default="MMD", help="linear")
    # parser.add_argument("--transform_method", type=str, default="class_linear", help="linear")
    # parser.add_argument("--transform_method", type=str, default="none", help="linear")
    parser.add_argument("--transform_epoch", type=int, default=10000, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_epoch", type=int, default=1000, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_lr", type=float, default=0.1, help="learning rate for transform (only for linear now)")
    parser.add_argument("--transform_lr", type=float, default=1e-6, help="learning rate for transform (only for linear now)")
    parser.add_argument("--transform_batch_size", type=int, default=256, help="batch size for transform (only for linear now)")

    parser.add_argument("--candidate_num", type=int, default=10, help="the candidate_num is the m in the m@recall")
    parser.add_argument("--search_range", type=int, default=[10, 20, 50, 100, 500, 1000], help="the search range for faiss search")
    # overlap_ratio
    parser.add_argument("--overlap_ratio", type=float, default=0.3, help="overlap ratio")

    parser.add_argument("--device", type=int, default=0, help="device for training")
    args = parser.parse_args()
    print(args)
    ans = main(args)
    print(ans)