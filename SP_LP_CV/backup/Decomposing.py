import torch
import numpy as np
import faiss
import os


# 加载特征数据
resnet_features = torch.load('./saved_features/resnet_feature.pt', map_location=torch.device('cpu'))
vgg_features = torch.load('./saved_features/vgg_features.pt', map_location=torch.device('cpu'))

# (60000, 512, 7, 7) -> (60000, 512*7*7)
resnet_features = resnet_features.view(resnet_features.size(0), -1)
vgg_features = vgg_features.view(vgg_features.size(0), -1)

# 将特征数据转换为numpy数组
resnet_features_np = resnet_features.numpy()
vgg_features_np = vgg_features.numpy()

# 通过PCA降到128维
from sklearn.decomposition import PCA

# 初始化PCA，设置目标维度为128
pca = PCA(n_components=128)

# 对resnet_features进行PCA降维
resnet_features_np = resnet_features_np.reshape(resnet_features_np.shape[0], -1)  # 确保数据是2D的
resnet_features_reduced = pca.fit_transform(resnet_features_np)

# 对vgg_features进行PCA降维
vgg_features_np = vgg_features_np.reshape(vgg_features_np.shape[0], -1)  # 确保数据是2D的
vgg_features_reduced = pca.fit_transform(vgg_features_np)

# 将降维后的特征数据保存(直接保存numpy数组)
np.save('./saved_features/resnet_feature_reduced.npy', resnet_features_reduced)
np.save('./saved_features/vgg_feature_reduced.npy', vgg_features_reduced)

