import numpy as np

def cal_recall(ans1, ans2):
    indices_ans1 = ans1[1]  # ans1的索引数组
    indices_ans2 = ans2[1]  # ans2的索引数组

    # 计算召回率
    # 初始化一个计数器来记录ans2中找到的与ans1相关的项数
    relevant_found = 0

    # 遍历ans1的索引数组中的每个查询的结果
    for query_idx in range(indices_ans1.shape[0]):
        # 获取当前查询对应的ans1和ans2的索引集合
        set_ans1 = set(indices_ans1[query_idx])
        set_ans2 = set(indices_ans2[query_idx])

        # 计算交集的大小，即ans2中找到的与ans1相关的项数
        relevant_found += len(set_ans1.intersection(set_ans2))

    # 计算总的相关项数，即所有查询在ans1中的总项数
    total_relevant = indices_ans1.size  # 这里假设每个查询返回的项数相同

    # 计算召回率
    recall = relevant_found / total_relevant
    return recall

def select_random_features(feature_matrix1, feature_matrix2, proportion):
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
    selected_indices = np.random.choice(all_indices, n_select, replace=False)
    
    # 返回选中的特征子集
    return feature_matrix1[selected_indices], feature_matrix2[selected_indices]
