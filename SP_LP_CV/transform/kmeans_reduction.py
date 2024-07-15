import torch
import numpy as np


def KMeansReduction(args, features:list):
    # 将特征数据转换为numpy数组
    # 通过PCA降到128维
    from sklearn.decomposition import PCA

    # 初始化PCA，设置目标维度为128
    pca = PCA(n_components=args.reduced_dim)
    
    features_reduced = [[] for _ in range(len(features))]
    for i in range(len(features)):
        # 对feature进行PCA降维
        features_reduced[i] = pca.fit_transform(features[i])

    return features_reduced
