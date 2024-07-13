import torch
import time
from torch import nn, optim
import torchvision
import sys
import gzip
import numpy as np
import struct
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image  # 导入 PIL 模块

# 定义VGG各种不同的结构和最后的全连接层结构
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'FC':    [512 * 7 * 7, 4096, 10]
}

# 将数据展开成二维数据，用在全连接层之前和卷积层之后
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.VGG_layer = self.vgg_block(cfg[vgg_name])
        self.FC_layer = self.fc_block(cfg['FC'])

    def forward(self, x):
        out_vgg = self.VGG_layer(x)
        out = out_vgg.view(out_vgg.size(0), -1)
        out = self.FC_layer(out)
        return out

    def vgg_block(self, cfg_vgg):
        layers = []
        in_channels = 1
        for out_channels in cfg_vgg:
            if out_channels == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def fc_block(self, cfg_fc):
        fc_net = nn.Sequential()
        fc_features, fc_hidden_units, fc_output_units = cfg_fc[0:]
        fc_net.add_module("fc", nn.Sequential(
            FlattenLayer(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_output_units)
        ))
        return fc_net

# 自定义数据集加载器
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image.squeeze(), mode='L')  # 转换为 PIL 图像
        if self.transform:
            image = self.transform(image)
        return image, label

    def load_images(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            print(f"Loaded {num} images with size {rows}x{cols} from {path}")
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols).astype(np.float32)
            images /= 255.0
        return images

    def load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            print(f"Loaded {num} labels from {path}")
            labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
        return labels

# 测试准确率
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                output = net(X.to(device))
                preds = output.argmax(dim=1)
                acc_sum += (preds == y.to(device)).float().sum().cpu().item()
                print(f"Batch Accuracy: {(preds == y.to(device)).float().mean().item():.4f}")
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 模型训练，定义损失函数、优化函数
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_idx, (X, y) in enumerate(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {l.item():.4f}, Batch Accuracy: {(y_hat.argmax(dim=1) == y).float().mean().item():.4f}")
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / len(train_iter), train_acc_sum / n, test_acc, time.time() - start))

def main():
    net = VGG('VGG16')
    print(net)

    # 一个batch_size为64张图片，进行梯度下降更新参数
    batch_size = 64
    # 使用cuda来训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载MNIST数据集，返回训练集和测试集
    train_images_path = r'C:\Users\liu18\image-work\imagetesting\MNIST-master\train-images-idx3-ubyte.gz'
    train_labels_path = r'C:\Users\liu18\image-work\imagetesting\MNIST-master\train-labels-idx1-ubyte.gz'
    test_images_path = r'C:\Users\liu18\image-work\imagetesting\MNIST-master\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'C:\Users\liu18\image-work\imagetesting\MNIST-master\t10k-labels-idx1-ubyte.gz'

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform)
    test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lr, num_epochs = 0.001, 5
    # 使用Adam优化算法替代传统的SGD，能够自适应学习率
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 训练--迭代更新参数
    train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

main()
