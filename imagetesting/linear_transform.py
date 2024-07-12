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
            nn.Linear(input_dim, 512),  # 第一层线性转换，将输入维度转换为512维
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(512, 256),  # 第二层线性转换，将512维转换为256维
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(256, output_dim)  # 第三层线性转换，将256维转换为输出维度
        )
    
    def forward(self, x):
        return self.align(x)
    
    def to(self, device):
        self.align.to(device)
        return self

def LinearTransform(args, origin: np.ndarray, target: np.ndarray, transformed_feature=None, test_origin=None, test_target=None):
    vgg_features = origin
    resnet_features = target
    
    if transformed_feature is None:
        transformed_feature = origin

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    vgg_features = torch.from_numpy(vgg_features).float().to(device)
    resnet_features = torch.from_numpy(resnet_features).float().to(device)
    if isinstance(transformed_feature, np.ndarray):
        transformed_feature = torch.from_numpy(transformed_feature).float().to(device)

    input_dim = vgg_features.shape[1]
    output_dim = resnet_features.shape[1]
    model = FeatureAlignNet(input_dim, output_dim)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.transform_lr)

    dataset = TensorDataset(vgg_features, resnet_features)
    dataloader = DataLoader(dataset, batch_size=args.transform_batch_size, shuffle=True)
    num_epochs = args.transform_epoch
    for epoch in trange(num_epochs):
        loss_avg = 0.0
        for vgg_batch, resnet_batch in dataloader:
            aligned_features = model(vgg_batch)
            loss = criterion(aligned_features, resnet_batch)
            loss_avg += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        wandb.log({"loss": loss_avg / len(dataloader), "epoch": epoch})

        if epoch % 100 == 0:
            if test_origin is not None and test_target is not None:
                test_aligned_features = model(torch.from_numpy(test_origin).float().to(device))
                test_loss = criterion(test_aligned_features, torch.from_numpy(test_target).float().to(device))
                wandb.log({"test_loss": test_loss.item()})
                print(f'Test Loss: {test_loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg / len(dataloader):.4f}')

    aligned_features = model(transformed_feature)
    return aligned_features.detach().cpu().numpy()
