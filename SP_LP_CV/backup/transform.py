import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np

# 假设vgg_features和resnet_features是你的特征矩阵，形状分别为[N, D_vgg]和[N, D_resnet]
# N是样本数量，D_vgg和D_resnet是特征维度
vgg_features = np.load('./saved_features/vgg_feature_reduced.npy')
resnet_features = np.load('./saved_features/resnet_feature_reduced.npy')

vgg_features = torch.from_numpy(vgg_features).float()
resnet_features = torch.from_numpy(resnet_features).float()

class FeatureAlignNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureAlignNet, self).__init__()
        self.align = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.align(x)

# 初始化模型
# 假设我们要将VGG特征转换为与ResNet特征空间对齐
input_dim = vgg_features.shape[1]  # VGG特征的维度
output_dim = resnet_features.shape[1]  # ResNet特征的维度
model = FeatureAlignNet(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import TensorDataset, DataLoader

# 创建TensorDataset
dataset = TensorDataset(vgg_features, resnet_features)

# 创建DataLoader
batch_size = 32  # 你可以根据你的硬件和需求来调整这个值
shuffle = True  # 如果你想在每个epoch后洗牌数据，将这个值设为True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
num_epochs = 1000
for epoch in trange(num_epochs):
    loss_avg = 0.0
    for vgg_batch, resnet_batch in dataloader:
        # 前向传播
        aligned_features = model(vgg_batch)
        
        # 计算损失
        loss = criterion(aligned_features, resnet_batch)
        loss_avg += loss.item()
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg/len(dataloader):.4f}')

aligned_features = model(vgg_features)
np.save(f'./saved_features/vgg_feature_aligned_{num_epochs}.npy', aligned_features.detach().numpy())
