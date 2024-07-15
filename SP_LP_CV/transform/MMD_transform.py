import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import wandb
import numpy as np

from .linear_transform import FeatureAlignNet
# class FeatureAlignNet(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FeatureAlignNet, self).__init__()
#         self.align = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )
    
#     def forward(self, x):
#         return self.align(x)
    
#     def to(self, device):
#         self.align.to(device)
#         return self

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算


# def MMDTransform(args, origin:np.ndarray, target:np.ndarray, transformed_feature1=None, transformed_feature2=None, test_origin=None, test_target=None):
#     # 均一化向量
#     vgg_features = origin
#     resnet_features = target
    
#     if transformed_feature1 is None:
#         transformed_feature1 = origin
#     if transformed_feature2 is None:
#         transformed_feature2 = target

#     if torch.cuda.is_available():
#         device = torch.device(f'cuda:{args.device}')
#     else:
#         device = torch.device('cpu')

#     vgg_features = torch.from_numpy(vgg_features).float().to(device)
#     resnet_features = torch.from_numpy(resnet_features).float().to(device)
#     if isinstance(transformed_feature1, np.ndarray):
#         transformed_feature1 = torch.from_numpy(transformed_feature1).float().to(device)
#     if isinstance(transformed_feature2, np.ndarray):
#         transformed_feature2 = torch.from_numpy(transformed_feature2).float().to(device)

#     # 初始化模型
#     # 假设我们要将VGG特征转换为与ResNet特征空间对齐
#     input_dim = vgg_features.shape[1]  # VGG特征的维度
#     output_dim = resnet_features.shape[1]  # ResNet特征的维度
#     # shuffle resnet_features
#     # np.random.shuffle(resnet_features)
#     # random_index = np.random.permutation(len(resnet_features))
#     # resnet_features = resnet_features[random_index]
#     model = FeatureAlignNet(input_dim, output_dim)
#     model.to(device)

#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=args.transform_lr)

#     from torch.utils.data import TensorDataset, DataLoader

#     # 创建TensorDataset
#     dataset = TensorDataset(vgg_features, resnet_features)

#     # 创建DataLoader
#     batch_size = args.transform_batch_size  # 你可以根据你的硬件和需求来调整这个值
#     shuffle = True  # 如果你想在每个epoch后洗牌数据，将这个值设为True
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     num_epochs = args.transform_epoch
#     for epoch in trange(num_epochs):
#         loss_avg = 0.0
#         for vgg_batch, resnet_batch in dataloader:
#             # 前向传播
#             # aligned_features = model(vgg_batch)
#             vgg_transformed = model(vgg_batch)
#             resnet_transformed = model(resnet_batch)
            
#             # 计算损失
#             # loss = criterion(aligned_features, resnet_batch)
#             # loss = torch.nn.functional.mse_loss(aligned_features, resnet_batch, reduction='sum')
#             loss = mmd_rbf(vgg_transformed, resnet_transformed)
#             loss_avg += loss.item()
            
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         wandb.log({"loss": loss_avg/len(dataloader), "epoch": epoch})

#         if epoch % 10 == 0:
#             if test_origin is not None and test_target is not None:
#                 test_aligned_features = model(torch.from_numpy(test_origin).float().to(device))
#                 # test_loss = criterion(test_aligned_features, torch.from_numpy(test_target).float().to(device))
#                 test_loss = torch.nn.functional.mse_loss(test_aligned_features, torch.from_numpy(test_target).float().to(device), reduction='sum')
#                 wandb.log({"test_loss": test_loss.item()})
#                 print(f'Test Loss: {test_loss.item():.4f}')
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg/len(dataloader):.4f}')
        

#     aligned_features1 = model(transformed_feature1)
#     aligned_features2 = model(transformed_feature2)
#     return aligned_features1.detach().cpu().numpy(), aligned_features2.detach().cpu().numpy()

def MMDTransform(args, origin, target, transformed_candidate=None, test_origin=None, test_target=None):
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
    model = FeatureAlignNet(input_dim, output_dim)
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
            # loss = criterion(aligned_features, target_batch)
            loss = mmd_rbf(aligned_features, target_batch)
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

