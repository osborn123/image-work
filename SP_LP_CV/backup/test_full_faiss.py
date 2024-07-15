import numpy as np
import faiss
import torch
from utils import cal_recall
# resnet_features_reduced = np.load('./saved_features/resnet_feature_reduced.npy')
# vgg_features_reduced = np.load('./saved_features/vgg_feature_reduced.npy')
resnet_features = torch.load('./saved_features/resnet_feature.pt', map_location='cpu')
vgg_features = torch.load('./saved_features/vgg_feature.pt', map_location='cpu')

resnet_features = resnet_features.numpy().reshape(resnet_features.shape[0], -1)
vgg_features = vgg_features.numpy().reshape(vgg_features.shape[0], -1)

# 获取特征的维度
dimension_resnet = resnet_features.shape[1]
dimension_vgg = vgg_features.shape[1]

# 初始化两个索引
index_resnet = faiss.IndexFlatL2(dimension_resnet)
index_vgg = faiss.IndexFlatL2(dimension_vgg)

# 将特征添加到对应的索引中
index_resnet.add(resnet_features)
index_vgg.add(vgg_features)

# search 返回的是距离和索引
ans1 = index_resnet.search(resnet_features[:5], 1000)

ans2 = index_vgg.search(resnet_features[:5], 1000)
print(cal_recall(ans1, ans2))