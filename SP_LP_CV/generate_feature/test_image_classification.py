
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
from sklearn.metrics import accuracy_score
# # 设置随机种子以保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# transform = transforms.Compose([
#     transforms.Resize(299),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# class VGGNetSimple(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGGNetSimple, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc_features = nn.Sequential(
#             nn.Linear(128 * 37 * 37, 256),  # 299 -> 149 -> 74 -> 37
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5)
#         )
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_features(x)
#         x = self.fc(x)
#         return x
    
#     def generate_feature(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         features = self.fc_features(x)
#         return features

# def ResidualBlock(in_channels, out_channels, stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect'),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         nn.BatchNorm2d(out_channels)
#     )

# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, padding_mode='reflect')
#         self.bn1 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.layer1 = self._make_layer(64, 64, 3)
#         self.layer2 = self._make_layer(64, 128, 4, stride=2)
#         self.layer3 = self._make_layer(128, 256, 6, stride=2)
#         self.layer4 = self._make_layer(256, 512, 3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, 10)
    
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(ResidualBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         x = self.pool1(self.bn1(F.relu(self.conv1(x))))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
    
#     def generate_feature(self, x):
#         x = self.pool1(self.bn1(F.relu(self.conv1(x))))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         return x

# def train_model(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(f'训练周期: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

# def train(model_name, train_loader, test_loader, gen_feature=False, verbose=False):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     if model_name == "ResNet":
#         model = ResNet().to(device)
#     elif model_name == "VGGNet":
#         model = VGGNetSimple().to(device)
#     else:
#         raise ValueError("不支持的模型名称")
    
#     model_path = f"./saved_model/image_classification/{model_name.lower()}_model.pth"
#     model_path = f"./saved_model/image_classification/{model_name.lower()}_model.pth"
#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path))
#         print(f"从磁盘加载{model_name}模型")
#     else:
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)
#         num_epochs = 3
        
#         if verbose:
#             for epoch in range(1, num_epochs + 1):
#                 train_model(model, device, train_loader, optimizer, criterion, epoch)
        
#         torch.save(model.state_dict(), model_path)
#         print(f"模型{model_name}保存到磁盘")
    
#     if gen_feature:
#         feature = cat_feature(model, test_loader)
#     else:
#         feature = None
    
#     return model, feature

# def cat_feature(model, data):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     with torch.no_grad():
#         features = []
#         for images, labels in tqdm.tqdm(data, desc="生成特征中"):
#             images = images.to(device)
#             feature = model.generate_feature(images)
#             features.append(feature.cpu())  # 将数据移到CPU
#         return torch.cat(features, dim=0)

# def test(model, data, feature):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     feature = feature.to(device)
#     model.to(device)
#     model.eval()
#     from sklearn.metrics import accuracy_score
#     with torch.no_grad():
#         outputs = model.fc(feature).argmax(dim=1)
#     acc = accuracy_score(data.targets.cpu().numpy(), outputs.cpu().numpy())
#     return acc

# def save_model(model, path):
#     torch.save(model.state_dict(), path)

# def load_model(model, path):
#     model.load_state_dict(torch.load(path))
#     return model
# vggmodel, _ = train(model_name="VGGNet", train_loader=train_loader, test_loader=test_loader, verbose=True)
# resmodel, _ = train(model_name="ResNet", train_loader=train_loader, test_loader=test_loader, verbose=True)


# if __name__ == "__main__":
#     if not os.path.exists("saved_model/image_classification"):
#         os.makedirs("saved_model/image_classification")

#     if not os.path.exists("saved_features/image_classification"):
#         os.makedirs("saved_features/image_classification")



#     vgg_feature = cat_feature(vggmodel, test_loader)
#     res_feature = cat_feature(resmodel, test_loader)

#     torch.save(vgg_feature, "saved_features/image_classification/MNIST_test_vggNet.pt")
#     torch.save(res_feature, "saved_features/image_classification/MNIST_test_resNet.pt")

#     print(test(resmodel, test_dataset, res_feature))
#     print(test(vggmodel, test_dataset, vgg_feature))






torch.manual_seed(42)
np.random.seed(42)

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# 数据集加载
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

labels = train_dataset.targets.numpy()
class_weights = np.bincount(labels)
class_weights = 1. / class_weights
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 模型定义
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
        self.fc_features = nn.Sequential(
            nn.Linear(128 * 37 * 37, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, num_classes)
        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x)
        x = self.fc(x)
        return x
    
    def generate_feature(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc_features(x)
        return features
    
    def _initialize_weights(self):
        # 使用Kaiming He初始化方法初始化卷积层和全连接层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class VGGNetNew(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNetNew, self).__init__()
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
        self.fc_features = nn.Sequential(
            nn.Linear(128 * 37 * 37, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x)
        x = self.fc(x)
        return x
    
    def generate_feature(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc_features(x)
        return features
    
    def _initialize_weights(self):
        # 使用Xavier初始化方法初始化卷积层和全连接层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def train_model(model, device, train_loader, optimizer, criterion, epoch):
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
    
    if model_name == "VGGNet":
        model = VGGNetSimple().to(device)
    elif model_name == "VGGNetNew":
        model = VGGNetNew().to(device)
    else:
        raise ValueError("不支持的模型名称")
    
    model_path = f"./saved_model/image_classification/{model_name.lower()}_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(device)  # 确保模型在正确的设备上
        print(f"从磁盘加载{model_name}模型")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 3
        
        if verbose:
            for epoch in range(1, num_epochs + 1):
                train_model(model, device, train_loader, optimizer, criterion, epoch)
        
        torch.save(model.state_dict(), model_path)
        print(f"模型{model_name}保存到磁盘")
    
    if gen_feature:
        feature = cat_feature(model, test_loader)
    else:
        feature = None
    
    return model, feature

def cat_feature(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        features = []
        for images, labels in tqdm.tqdm(data_loader, desc="Generating features"):
            images = images.to(device)
            feature = model.generate_feature(images)
            features.append(feature.cpu())  # Move to CPU before appending
        return torch.cat(features, dim=0)

def test(model, data, feature):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature = feature.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.fc(feature).argmax(dim=1)
    acc = accuracy_score(data.targets.cpu().numpy(), outputs.cpu().numpy())
    return acc

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

vgg_model_path = "saved_model/image_classification/vgg_model.pth"
vgg_new_model_path = "saved_model/image_classification/vgg_new_model.pth"

vggmodel, _ = train(model_name="VGGNet", train_loader=train_loader, test_loader=test_loader, verbose=True)
vggnewmodel, _ = train(model_name="VGGNetNew", train_loader=train_loader, test_loader=test_loader, verbose=True)
if not os.path.exists("saved_model/image_classification"):
    os.makedirs("saved_model/image_classification")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(vgg_model_path):
        print("Loading existing VGGNet model...")
        vggmodel = VGGNetSimple()
        vggmodel = load_model(vggmodel, vgg_model_path)
        vggmodel = vggmodel.to(device)  # 确保模型在正确的设备上
    else:
        print("Training new VGGNet model...")
        vggmodel, _ = train(model_name="VGGNet", train_loader=train_loader, test_loader=test_loader, verbose=True)
        save_model(vggmodel, vgg_model_path)

    if os.path.exists(vgg_new_model_path):
        print("Loading existing VGGNetNew model...")
        vggnewmodel = VGGNetNew()
        vggnewmodel = load_model(vggnewmodel, vgg_new_model_path)
        vggnewmodel = vggnewmodel.to(device)  # 确保模型在正确的设备上
    else:
        print("Training new VGGNetNew model...")
        vggnewmodel, _ = train(model_name="VGGNetNew", train_loader=train_loader, test_loader=test_loader, verbose=True)
        save_model(vggnewmodel, vgg_new_model_path)

    if not os.path.exists("saved_features/image_classification"):
        os.makedirs("saved_features/image_classification")

    vgg_feature = cat_feature(vggmodel, test_loader)
    vgg_new_feature = cat_feature(vggnewmodel, test_loader)

    torch.save(vgg_feature, "saved_features/image_classification/MNIST_test_vggNet.pt")
    torch.save(vgg_new_feature, "saved_features/image_classification/MNIST_test_vggNetNew.pt")

    print(f"VGGNet Accuracy: {test(vggmodel, test_dataset, vgg_feature)}")
    print(f"VGGNetNew Accuracy: {test(vggnewmodel, test_dataset, vgg_new_feature)}")