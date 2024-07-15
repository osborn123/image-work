import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import wandb

# class FeatureAlignNet(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FeatureAlignNet, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)

# class FeatureAlignNet(nn.Module):
#     def __init__(self, args, input_dim, output_dim):
#         super(FeatureAlignNet, self).__init__()
#         layers = args.layers
#         with_bias = args.with_bias
#         with_activation = args.with_activation
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )

#     def forward(self, x):
#         return self.network(x)

import torch.nn as nn

class FeatureAlignNet(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(FeatureAlignNet, self).__init__()
        layers = args.layers  # number of layers
        with_bias = True if args.with_bias == 1 else False
        with_activation = True if args.with_activation == 1 else False
        
        # 创建层列表
        network_layers = []
        
        # 添加第一层
        current_dim = input_dim
        next_dim = 1024 if input_dim > 1024 else output_dim
        network_layers.append(nn.Linear(current_dim, next_dim, bias=with_bias))
        if with_activation and layers > 1:
            network_layers.append(nn.ReLU())
        
        # 添加隐藏层
        for _ in range(1, layers-1):
            current_dim = next_dim
            next_dim = max(next_dim // 2, output_dim)
            network_layers.append(nn.Linear(current_dim, next_dim, bias=with_bias))
            if with_activation:
                network_layers.append(nn.ReLU())
        
        # 添加输出层
        if layers > 1:
            network_layers.append(nn.Linear(next_dim, output_dim, bias=with_bias))
        
        # 创建顺序层
        self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        return self.network(x)


def LinearTransform(args, origin, target, transformed_candidate=None, test_origin=None, test_target=None):
    # 将输入数据转换为Tensor并移动到指定设备
    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).float().to(args.device)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float().to(args.device)

    if transformed_candidate is not None and isinstance(transformed_candidate, np.ndarray):
        transformed_candidate = torch.from_numpy(transformed_candidate).float().to(args.device)
    if test_origin is not None and isinstance(test_origin, np.ndarray):
        test_origin = torch.from_numpy(test_origin).float().to(args.device)
    if test_target is not None and isinstance(test_target, np.ndarray):
        test_target = torch.from_numpy(test_target).float().to(args.device)

    # 定义输入和输出维度
    input_dim, output_dim = origin.shape[1], target.shape[1]

    # 初始化模型，损失函数和优化器
    model = FeatureAlignNet(args, input_dim, output_dim)
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.transform_lr)

    # 创建DataLoader
    dataset = TensorDataset(origin, target)
    dataloader = DataLoader(dataset, batch_size=args.transform_batch_size, shuffle=True)

    # 训练模型
    for epoch in trange(args.transform_epoch):
        model.train()
        loss_avg = 0.0
        for origin_batch, target_batch in dataloader:
            # 前向传播
            aligned_features = model(origin_batch)

            # 计算损失
            loss = criterion(aligned_features, target_batch)
            loss_avg += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 记录和打印损失
        wandb.log({"loss": loss_avg / len(dataloader), "epoch": epoch})

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                if test_origin is not None and test_target is not None:
                    test_aligned_features = model(test_origin)
                    test_loss = criterion(test_aligned_features, test_target)
                    wandb.log({"test_loss": test_loss.item()})
                    print(f'Test Loss: {test_loss.item():.4f}')
            print(f'Epoch [{epoch + 1}/{args.transform_epoch}], Loss: {loss_avg / len(dataloader):.4f}')

    # 转换特征
    model.eval()
    with torch.no_grad():
        if transformed_candidate is not None:
            aligned_features = model(transformed_candidate)
            return aligned_features.detach().cpu().numpy()
        else:
            return None
