import sys
import os
import gzip
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from resnet import ResNet  # 调整导入路径

# 检查 GPU 是否可用
print("CUDA Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count: ", torch.cuda.device_count())
    print("Current device: ", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

# 加载 MNIST 数据集
def load_mnist_images(filename):
    with gzip.open(filename, 'r') as f:
        f.read(16)  # Skip the magic number and dimensions
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 28, 28, 1)
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'r') as f:
        f.read(8)  # Skip the magic number and dimensions
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
        return labels

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和测试集
train_images = load_mnist_images(r'C:\Users\liu18\image-work\imagetesting\MNIST-master\train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels(r'C:\Users\liu18\image-work\imagetesting\MNIST-master\train-labels-idx1-ubyte.gz')
test_images = load_mnist_images(r'C:\Users\liu18\image-work\imagetesting\MNIST-master\t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels(r'C:\Users\liu18\image-work\imagetesting\MNIST-master\t10k-labels-idx1-ubyte.gz')

# 转换为 PyTorch 张量并创建 DataLoader
train_images_tensor = torch.tensor(train_images).permute(0, 3, 1, 2).repeat(1, 3, 1, 1)  # 重复通道以匹配3通道输入
train_labels_tensor = torch.tensor(train_labels)
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_images_tensor = torch.tensor(test_images).permute(0, 3, 1, 2).repeat(1, 3, 1, 1)  # 重复通道以匹配3通道输入
test_labels_tensor = torch.tensor(test_labels)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# PyTorch 模型训练
def train_pytorch_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()  # 记录每个 epoch 的开始时间
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:  # 每 100 个批次打印一次日志
                print(f"Batch {i}, Loss: {loss.item()}")
        end_time = time.time()  # 记录每个 epoch 的结束时间
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Time: {epoch_time:.2f} seconds")

# PyTorch 模型评估
def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"PyTorch Model Accuracy: {100 * correct / total}%")

# 初始化模型、损失函数和优化器
pytorch_model = ResNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# 将模型和数据移至 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练和评估 PyTorch 模型
train_pytorch_model(pytorch_model, train_loader, criterion, optimizer, device, epochs=5)
evaluate_pytorch_model(pytorch_model, test_loader, device)

# 保存模型参数
torch.save(pytorch_model.state_dict(), 'pytorch_model.pth')

# 加载模型参数
pytorch_model.load_state_dict(torch.load('pytorch_model.pth'))

# 评估加载后的模型
evaluate_pytorch_model(pytorch_model, test_loader, device)
