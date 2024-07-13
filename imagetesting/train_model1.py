import sys
import os
import gzip
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from resnet import ResNet  

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
        data = data / 127.5 - 1  # 将数据范围从 [0, 255] 缩放到 [-1, 1]
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'r') as f:
        f.read(8)  # Skip the magic number and dimensions
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
        return labels

# 计算数据集的均值和标准差
def compute_mean_std(images):
    mean = np.mean(images)
    std = np.std(images)
    return mean, std

# 数据预处理
data_dir = os.path.join(os.path.dirname(__file__), 'MNIST-master')
train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

# 计算训练集的均值和标准差
mean, std = compute_mean_std(train_images)

# 转换为 PyTorch 张量并创建 DataLoader
transform = transforms.Compose([
    transforms.Normalize((mean,), (std,))
])

train_images_tensor = transform(torch.tensor(train_images).permute(0, 3, 1, 2).repeat(1, 3, 1, 1))  # 复制到3个通道
train_labels_tensor = torch.tensor(train_labels)
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)

test_images_tensor = transform(torch.tensor(test_images).permute(0, 3, 1, 2).repeat(1, 3, 1, 1))  # 复制到3个通道
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

# Top-k 评估函数
def topk_test(model, device, test_loader, k=5):
    model.eval()
    model.to(device)
    topk_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
            topk_correct += sum([labels[i] in pred[i] for i in range(len(labels))])

    topk_accuracy = 100. * topk_correct / len(test_loader.dataset)
    print(f'\nTest set: Top-{k} Accuracy: {topk_correct}/{len(test_loader.dataset)} ({topk_accuracy:.0f}%)\n')
    return topk_accuracy

# 初始化模型、损失函数和优化器
pytorch_model = ResNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# 将模型和数据移至 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义不同的重叠率和 search range 预设
overlap_presets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
search_range_presets = [10, 20, 50, 100]

# 训练和评估每个重叠率和 search range 下的 PyTorch 模型
for overlap_rate in overlap_presets:
    # 根据重叠率划分训练集
    num_samples = len(train_dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * num_samples))
    overlap_count = int(split * overlap_rate)
    
    train_indices = indices[:split - overlap_count]
    val_indices = indices[split - overlap_count:]
    overlap_indices = val_indices[:overlap_count]

    final_train_indices = np.concatenate((train_indices, overlap_indices))
    train_loader = DataLoader(Subset(train_dataset, final_train_indices), batch_size=64, shuffle=True)
    
    print(f"\nTraining with overlap rate: {overlap_rate*100}%")
    
    # 训练 PyTorch 模型
    train_pytorch_model(pytorch_model, train_loader, criterion, optimizer, device, epochs=5)
    
    # 评估 PyTorch 模型
    evaluate_pytorch_model(pytorch_model, test_loader, device)
    
    # 执行 top-k 测试
    for search_range in search_range_presets:
        print(f"\nTesting with search range: {search_range}")
        topk_test(pytorch_model, device, test_loader, k=search_range)
