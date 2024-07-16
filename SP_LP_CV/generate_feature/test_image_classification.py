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
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
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

# 定义适应 299x299 输入大小的网络结构
class VGGNetSimple(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNetSimple, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 根据 299x299 输入调整全连接层输入大小
        self.fc_features = nn.Sequential(
            nn.Linear(128 * 37 * 37, 256),  # 299 -> 149 -> 74 -> 37
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x= self.fc_features(x)
        x = self.fc(x)
        return x
    
    def generate_feature(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc_features(x)
        return features
# Example usage:
# model = VGGNet(num_classes=10)
# output = model(torch.randn(1, 3, 224, 224))



def ResidualBlock(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels)
    )

class ResNet(nn.Module):
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

def train_zfnet(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def train(model_name, train_loader, test_loader, gen_feature=False, verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if model_name == "ZFNet":
        model = ZFNet().to(device)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 3
        if os.path.exists(f"./saved_features/image_classification/{model_name.lower()}_mnist.pth"):
            model.load_state_dict(torch.load(f"./saved_features/image_classification/{model_name.lower()}_mnist.pth"))
            print(f"Loaded model from disk")
        else:
            for epoch in range(1, num_epochs + 1):
                train_zfnet(model, device, train_loader, optimizer, criterion, epoch)
            torch.save(model.state_dict(), f"./saved_features/image_classification/{model_name.lower()}_mnist.pth")
        
                #test_zfnet(model, device, test_loader, criterion)
    elif model_name == "ResNet":
        model = ResNet().to(device)
    elif model_name == "VGGNet":
        model = VGGNetSimple().to(device)
    else:
        raise ValueError("model_name not supported")
    
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
            features.append(feature.cpu())  # Move to CPU before appending
        return torch.cat(features, dim=0)

def test(model, data, feature):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(data.targets, model.fc(feature).argmax(dim=1).cpu().numpy())
    return acc

# zfmodel, zf_feature = train(model_name="ZFNet", verbose=True)
vggmodel, vgg_feature = train(model_name="VGGNet", verbose=True)
resmodel, res_feature = train(model_name="ResNet", verbose=True)


if __name__ == "__main__":
    if not os.path.exists("saved_features/image_classification"):
        os.mkdir("saved_features/image_classification")
    # zf_feature = cat_feature(zfmodel, test_loader)
    vgg_feature = cat_feature(vggmodel, test_loader)
    res_feature = cat_feature(resmodel, test_loader)
    # torch.save(zf_feature, "saved_features/image_classification/MNIST_test_zfNet.pt")
    torch.save(vgg_feature, "saved_features/image_classification/MNIST_test_vggNet.pt")
    torch.save(res_feature, "saved_features/image_classification/MNIST_test_resNet.pt")
    print(test(resmodel, test_dataset, res_feature))
    print(test(vggmodel, test_dataset, vgg_feature))
