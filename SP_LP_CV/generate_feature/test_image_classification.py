import sys
import os
import gzip
import time
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class ZFNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ZFNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def generate_feature(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        return x


def ResidualBlock(in_channels, out_channels, stride=1):
    """
    A helper function to create a residual block for the ResNet architecture.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels)
    )

class ResNet(nn.Module):
    """
    A implementation of the ResNet architecture from the paper 'Deep Residual Learning for Image Recognition' by He et al.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def generate_feature(self, x):
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def train(model_name, gen_feature=False, verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if model_name == "ZFNet":
        model = ZFNet().to(device)
    elif model_name == "ResNet":
        model = ResNet().to(device)
    else:
        raise ValueError("model_name not supported")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if os.path.exists(f"./saved_features/image_classification/{model_name.lower()}_mnist.pth"):
        model.load_state_dict(torch.load(f"./saved_features/image_classification/{model_name.lower()}_mnist.pth"))
        print(f"Loaded model from disk")
    else:
        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        torch.save(model.state_dict(), f"./saved_features/image_classification/{model_name.lower()}_mnist.pth")
    if gen_feature:
        feature = cat_feature(model, test_loader)
    else:
        feature = None
    return model, feature

def cat_feature(model, data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        features = []
        for images, labels in tqdm.tqdm(data, desc="Generating features"):
            images = images.to(device)
            feature = model.generate_feature(images)
            features.append(feature)
        return torch.cat(features, dim=0)

zfmodel, zf_feature = train(model_name="ZFNet", verbose=True)
resmodel, res_feature = train(model_name="ResNet", verbose=True)

if __name__ == "__main__":
    if not os.path.exists("saved_features/image_classification"):
        os.mkdir("saved_features/image_classification")
    zf_feature = cat_feature(zfmodel, test_loader)
    res_feature = cat_feature(resmodel, test_loader)
    torch.save(zf_feature, "saved_features/image_classification/MNIST_test_zfNet.pt")
    torch.save(res_feature, "saved_features/image_classification/MNIST_test_resNet.pt")