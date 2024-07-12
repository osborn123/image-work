import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import numpy as np
import wandb

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
    
    def to(self, device):
        self.align.to(device)
        return self

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def MMDTransform(args, origin: np.ndarray, target: np.ndarray, transformed_feature1=None, transformed_feature2=None, test_origin=None, test_target=None):
    vgg_features = origin
    resnet_features = target
    
    if transformed_feature1 is None:
        transformed_feature1 = origin
    if transformed_feature2 is None:
        transformed_feature2 = target

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    vgg_features = torch.from_numpy(vgg_features).float().to(device)
    resnet_features = torch.from_numpy(resnet_features).float().to(device)
    if isinstance(transformed_feature1, np.ndarray):
        transformed_feature1 = torch.from_numpy(transformed_feature1).float().to(device)
    if isinstance(transformed_feature2, np.ndarray):
        transformed_feature2 = torch.from_numpy(transformed_feature2).float().to(device)

    input_dim = vgg_features.shape[1]
    output_dim = resnet_features.shape[1]
    model = FeatureAlignNet(input_dim, output_dim)
    model.to(device)

    criterion = mmd_rbf
    optimizer = optim.AdamW(model.parameters(), lr=args.transform_lr)

    dataset = TensorDataset(vgg_features, resnet_features)
    dataloader = DataLoader(dataset, batch_size=args.transform_batch_size, shuffle=True)
    num_epochs = args.transform_epoch
    for epoch in trange(num_epochs):
        loss_avg = 0.0
        for vgg_batch, resnet_batch in dataloader:
            vgg_transformed = model(vgg_batch)
            resnet_transformed = model(resnet_batch)
            loss = criterion(vgg_transformed, resnet_transformed)
            loss_avg += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        wandb.log({"loss": loss_avg / len(dataloader), "epoch": epoch})

        if epoch % 10 == 0:
            if test_origin is not None and test_target is not None:
                test_aligned_features = model(torch.from_numpy(test_origin).float().to(device))
                test_loss = criterion(test_aligned_features, torch.from_numpy(test_target).float().to(device))
                wandb.log({"test_loss": test_loss.item()})
                print(f'Test Loss: {test_loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg / len(dataloader):.4f}')

    aligned_features1 = model(transformed_feature1)
    aligned_features2 = model(transformed_feature2)
    return aligned_features1.detach().cpu().numpy(), aligned_features2.detach().cpu().numpy()
