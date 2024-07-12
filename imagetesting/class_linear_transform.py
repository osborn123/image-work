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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.align(x)
    
    def to(self, device):
        self.align.to(device)
        return self

def train_class_linear_transform(args, origin, target, labels, device):
    unique_labels = np.unique(labels)
    label_transformers = {}

    for label in unique_labels:
        mask = (labels == label)
        origin_label = origin[mask]
        target_label = target[mask]

        origin_label = torch.from_numpy(origin_label).float().to(device)
        target_label = torch.from_numpy(target_label).float().to(device)

        input_dim = origin_label.shape[1]
        output_dim = target_label.shape[1]
        model = FeatureAlignNet(input_dim, output_dim)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.transform_lr)

        dataset = TensorDataset(origin_label, target_label)
        dataloader = DataLoader(dataset, batch_size=args.transform_batch_size, shuffle=True)

        for epoch in trange(args.transform_epoch, desc=f"Training for label {label}"):
            loss_avg = 0.0
            for origin_batch, target_batch in dataloader:
                optimizer.zero_grad()
                aligned_features = model(origin_batch)
                loss = criterion(aligned_features, target_batch)
                loss_avg += loss.item()
                loss.backward()
                optimizer.step()

            wandb.log({f"loss_label_{label}": loss_avg / len(dataloader), "epoch": epoch})
            print(f"Label {label} - Epoch [{epoch+1}/{args.transform_epoch}], Loss: {loss_avg / len(dataloader):.4f}")

        label_transformers[label] = model

    return label_transformers

def ClassLinearTransform(args, origin, target, labels, transformed_feature=None):
    if transformed_feature is None:
        transformed_feature = origin

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    label_transformers = train_class_linear_transform(args, origin, target, labels, device)

    transformed_feature = torch.from_numpy(transformed_feature).float().to(device)
    labels = torch.from_numpy(labels).to(device)

    transformed_features = []
    for i in range(transformed_feature.shape[0]):
        label = labels[i].item()
        model = label_transformers[label]
        transformed_features.append(model(transformed_feature[i].unsqueeze(0)).cpu().numpy())

    return np.concatenate(transformed_features, axis=0)
