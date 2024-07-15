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

def cal_recall_st(args, feature_source, feature_target, only_2_index):
    # index_feature1 = faiss.IndexFlatL2(feature1.shape[1])
    # index_feature2 = faiss.IndexFlatL2(feature2.shape[1])
    index_source = faiss.IndexFlatL2(feature_source.shape[1])
    index_target = faiss.IndexFlatL2(feature_target.shape[1])

    logging.info("Adding features to index...")
    index_source.add(feature_source[only_2_index])
    index_target.add(feature_target[only_2_index])

    recall_list = []
    GT_source_list = []
    GT_target_list = []
    for search_range in args.search_range:
        source_list = index_source.search(feature_source[:args.candidate_num], search_range)
        target_list = index_target.search(feature_target[:args.candidate_num], search_range)

        GT_source_list.append(source_list)
        GT_target_list.append(target_list)

        recall_st = cal_recall(source_list, target_list)

        recall_list.append(recall_st)
        # print(f"{args.candidate_num}@{search_range} recall: {recall} ({cal_recall(ans1, ans3)})")
        print(f"{args.candidate_num}@{search_range} recall_st: {recall_st}")
    return GT_source_list, GT_target_list, recall_list

# def cal_recall_results(args, feature_transformed, feautre_target, query_node_transformed, query_node_target, only_2_index, type="rs"):
#     index_transformed = faiss.IndexFlatL2(feature_transformed.shape[1])
#     index_target = faiss.IndexFlatL2(feautre_target.shape[1])

#     logging.info("Adding features to index...")
#     index_transformed.add(feature_transformed[only_2_index])
#     index_target.add(feautre_target[only_2_index])

#     recall_list = []
#     for search_range in args.search_range:
#         recall = cal_recall(index_transformed.search(query_node_transformed, search_range), 
#                 index_target.search(query_node_target, search_range))
#         recall_list.append(recall)
#         print(f"{args.candidate_num}@{search_range} recall_{type}: {recall}")
#     return recall_list

def cal_recall_results(args, feature_transformed, query_node, target_gt_list, only_2_index, type="rs"):
    index_transformed = faiss.IndexFlatL2(feature_transformed.shape[1])
    index_transformed.add(feature_transformed[only_2_index])

    recall_list = []
    for i, search_range in enumerate(args.search_range):
        recall = cal_recall(index_transformed.search(query_node, search_range), target_gt_list[i])
        print(f"{args.candidate_num}@{search_range} recall_{type}: {recall}")
        recall_list.append(recall)
    return recall_list

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(299),  # 调整图像大小以适应InceptionV3的输入
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将灰度图像转换为彩色图像
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
label = train_dataset.targets.numpy()

def split_through_class(feature1, label, overlap_ratio:float=0.3):
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
    only_1_index, only_2_index, overlap_index = split_through_class(feature1, label, overlap_ratio=args.overlap_ratio)

    # cal_recall_st(args, feature1, feature2, only_2_index)
    recall_list = []
    GT_source_list, GT_target_list, recall_st = cal_recall_st(args, feature_source=feature2, feature_target=feature1, only_2_index=only_2_index)

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
        logging.info(f"overlap_ratio:{args.overlap_ratio} epochs:{args.transform_epoch} lr:{args.transform_lr} batch_size:{args.transform_batch_size}")
    elif args.transform_method == "class_linear":
        # for class_index in 
        # index_2 = np.concatenate([only_2_index, overlap_index])
        feature2_selected = feature2.copy()
        for class_index in np.unique(label):
            overlap_index_per_cls = overlap_index[label[overlap_index] == class_index]
            index_2_cls = only_2_index[label[only_2_index] == class_index]
            feature2_selected[index_2_cls] = LinearTransform(args, feature2[overlap_index_per_cls], feature1[overlap_index_per_cls], transformed_feature=feature2[index_2_cls], test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
    elif args.transform_method == "" or args.transform_method.lower() == "none":
        logging.info("No transform...")
        feature2_selected = feature2
    elif args.transform_method == "MMD":
        logging.info("Performing MMD transform...")
        feature2_selected = feature2.copy()
        feature1_overlapd = feature1[overlap_index]
        if args.MMD_shuffle:
            logging.info("Shuffling the overlap index...")
            np.random.shuffle(overlap_index)
        else:
            logging.info("Not shuffling the overlap index...")
        feature2_overlapd = feature2[overlap_index]
        # feature1, feature2_selected = MMDTransform(args, feature1, feature2_selected, transformed_feature1=feature1, transformed_feature2=feature2, test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
        feature1, feature2_selected = MMDTransform(args, feature2_overlapd, feature1_overlapd, transformed_feature1=feature1, transformed_feature2=feature2, test_origin=feature2[only_2_index], test_target=feature1[only_2_index])
        logging.info(f"overlap_ratio:{args.overlap_ratio} epochs:{args.transform_epoch} lr:{args.transform_lr} batch_size:{args.transform_batch_size}")

    else:
        raise ValueError("transform method not found")
    # def cal_recall_results(args, feature_transformed, query_node, target_gt_list, type="rs"):

    recall_list.append(cal_recall_results(args, feature2_selected, feature1[:args.candidate_num], GT_source_list, only_2_index, type="rs"))
    print("=======")
    recall_list.append(cal_recall_results(args, feature2_selected, feature1[:args.candidate_num], GT_target_list, only_2_index, type="rt"))

    # recall_list.append(cal_recall_results(args, feature2_selected, feature2, feature2[:args.candidate_num], only_2_index, type="rs"))
    # recall_list.append(cal_recall_results(args, feature2_selected, feature1, feature1[:args.candidate_num], only_2_index, type="rt"))

    # recall_list.append(cal_recall_results(args, feature2_selected, feature2, query_node_transformed=feature2_selected[:args.candidate_num], query_node_target=feature2[:args.candidate_num], only_2_index=only_2_index, type="rs"))
    # recall_list.append(cal_recall_results(args, feature2_selected, feature1, feature1[:args.candidate_num], feature1[:args.candidate_num], only_2_index, type="rt"))

    
    # logging.info("Creating index...")
    # index_feature1 = faiss.IndexFlatL2(feature1.shape[1])
    # index_feature2 = faiss.IndexFlatL2(feature2.shape[1])
    # index_feature3 = faiss.IndexFlatL2(feature2.shape[1])

    # logging.info("Adding features to index...")
    # # index_feature1.add(feature1[only_2_index])
    # index_feature1.add(feature1[only_2_index])
    # index_feature2.add(feature2_selected[only_2_index])
    # index_feature3.add(feature2[only_2_index])
    # # index_feature2.add(feature2[only_2_index])

    # recall_list = []
    # for search_range in args.search_range:
    #     logging.info("Performing search...")
    #     ans1 = index_feature1.search(feature1[:args.candidate_num], search_range)
    #     ans2 = index_feature2.search(feature1[:args.candidate_num], search_range)
    #     ans3 = index_feature3.search(feature2[:args.candidate_num], search_range)
        
    #     logging.info("Calculating recall...")
    #     recall = cal_recall(ans1, ans2)
    #     recall_list.append(recall)
    #     wandb.log({f"recall@{search_range}": recall})
    #     print(f"{args.candidate_num}@{search_range} recall: {recall} ({cal_recall(ans1, ans3)})")

    #     logging.info("Calculating recall...")
    #     recall = cal_recall(ans3, ans2)
    #     recall_list.append(recall)
    #     wandb.log({f"recall@{search_range}": recall})
    #     print(f"{args.candidate_num}@{search_range} recall: {recall} ({cal_recall(ans1, ans3)})")

    end_time = time.time()
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
    wandb.log({"time": end_time - start_time})
    return recall_list

if __name__ == "__main__":
    seed_everything(42)
    # 用parser引导参数
    parser = argparse.ArgumentParser()
    # parser.add_argument("--reduced_dim", type=int, default=128, help="0=>no reduction, other=> call kmeans to reduction")
    parser.add_argument("--reduced_dim", type=int, default=0, help="0=>no reduction, other=> call kmeans to reduction")

    parser.add_argument("--run_entity", type=str, default="", help="run entity")
    parser.add_argument("--mode", type=str, default="disabled", help="enabled or disabled")

    parser.add_argument("--saved_features_dir", type=str, default="saved_features", help="saved features dir")
    parser.add_argument("--feature1_path", type=str, default="inception_mnist_features.pt", help="feature1 path")
    parser.add_argument("--feature2_path", type=str, default="vgg_mnist_reduced_features.pt", help="feature2 path")
    parser.add_argument("--feature3_path", type=str, default="inception_mnist_features.pt", help="feature3 path")

    parser.add_argument("--select_method", type=str, default="random", help="random") # random full
    parser.add_argument("--select_proportion", type=float, default=0.001, help="select proportion")

    # parser.add_argument("--transform_method", type=str, default="linear", help="linear")
    parser.add_argument("--transform_method", type=str, default="MMD", help="linear")
    # parser.add_argument("--transform_method", type=str, default="None", help="linear")
    # MMD_shuffle
    parser.add_argument("--MMD_shuffle", type=int, default=0, help="linear")
    # parser.add_argument("--transform_method", type=str, default="class_linear", help="linear")
    # parser.add_argument("--transform_method", type=str, default="none", help="linear")
    # parser.add_argument("--transform_epoch", type=int, default=10000, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_epoch", type=int, default=0, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_epoch", type=int, default=1000, help="epoch for transform (only for linear now)")
    parser.add_argument("--transform_epoch", type=int, default=1000, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_epoch", type=int, default=0, help="epoch for transform (only for linear now)")
    # parser.add_argument("--transform_lr", type=float, default=0.1, help="learning rate for transform (only for linear now)")
    parser.add_argument("--transform_lr", type=float, default=1e-6, help="learning rate for transform (only for linear now)")
    parser.add_argument("--transform_batch_size", type=int, default=256, help="batch size for transform (only for linear now)")

    parser.add_argument("--candidate_num", type=int, default=10, help="the candidate_num is the m in the m@recall")
    parser.add_argument("--search_range", type=int, default=[10, 20, 50, 100, 500, 1000], help="the search range for faiss search")
    # overlap_ratio
    parser.add_argument("--overlap_ratio", type=float, default=0.3, help="overlap ratio")
    # parser.add_argument("--overlap_ratio", type=float, default=0.6, help="overlap ratio")

    parser.add_argument("--device", type=int, default=0, help="device for training")
    args = parser.parse_args()
    print(args)
    ans = main(args)
    print(ans)