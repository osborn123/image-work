import torch
from torchvision import models, transforms
from torchvision.models import VGG19_Weights, Inception_V3_Weights
import numpy as np
from PIL import Image
import gzip
import struct
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
inception_v3 = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True).to(device)

# 加载权重
vgg19_weights = torch.load('/Users/liuxuchen/imagetesting/model_weights/vgg19-dcbb9e9d.pth')
inception_v3_weights = torch.load('/Users/liuxuchen/imagetesting/model_weights/inception_v3_google-1a9a5a14.pth')

# 加载权重到模型
vgg19.load_state_dict(vgg19_weights)
inception_v3.load_state_dict(inception_v3_weights)

# 移除 InceptionV3 模型的辅助分类器
inception_v3.aux_logits = False
inception_v3.AuxLogits = None

# 定义预处理步骤
vgg19_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 将MNIST灰度图转换为RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inception_preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # 将MNIST灰度图转换为RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义MNIST数据集类
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def read_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        _ = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return data

def read_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        _ = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def train_model(model, optimizer, dataloader, criterion, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return accuracy

def main():
    # 读取训练集
    train_images = read_mnist_images('/Users/liuxuchen/imagetesting/MNIST-master/train-images-idx3-ubyte.gz')
    train_labels = read_mnist_labels('/Users/liuxuchen/imagetesting/MNIST-master/train-labels-idx1-ubyte.gz')
    
    # 读取测试集
    test_images = read_mnist_images('/Users/liuxuchen/imagetesting/MNIST-master/t10k-images-idx3-ubyte.gz')
    test_labels = read_mnist_labels('/Users/liuxuchen/imagetesting/MNIST-master/t10k-labels-idx1-ubyte.gz')
    
    # 创建数据集和数据加载器
    vgg19_train_dataset = MNISTDataset(train_images, train_labels, transform=vgg19_preprocess)
    vgg19_test_dataset = MNISTDataset(test_images, test_labels, transform=vgg19_preprocess)
    inception_train_dataset = MNISTDataset(train_images, train_labels, transform=inception_preprocess)
    inception_test_dataset = MNISTDataset(test_images, test_labels, transform=inception_preprocess)
    
    vgg19_train_dataloader = DataLoader(vgg19_train_dataset, batch_size=32, shuffle=True)
    vgg19_test_dataloader = DataLoader(vgg19_test_dataset, batch_size=32, shuffle=False)
    inception_train_dataloader = DataLoader(inception_train_dataset, batch_size=32, shuffle=True)
    inception_test_dataloader = DataLoader(inception_test_dataset, batch_size=32, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    vgg19_optimizer = optim.SGD(vgg19.parameters(), lr=0.001, momentum=0.9)
    inception_optimizer = optim.SGD(inception_v3.parameters(), lr=0.001, momentum=0.9)

    # 训练VGG19模型
    print("Training VGG19 model...")
    train_model(vgg19, vgg19_optimizer, vgg19_train_dataloader, criterion, device, epochs=5)
    
    # 训练InceptionV3模型
    print("Training InceptionV3 model...")
    train_model(inception_v3, inception_optimizer, inception_train_dataloader, criterion, device, epochs=5)
    
    # 测试VGG19模型
    print("Testing VGG19 model...")
    vgg19_accuracy = test_model(vgg19, vgg19_test_dataloader, device)
    print(f'VGG19 Test Accuracy: {vgg19_accuracy * 100:.2f}%')
    
    # 测试InceptionV3模型
    print("Testing InceptionV3 model...")
    inception_accuracy = test_model(inception_v3, inception_test_dataloader, device)
    print(f'InceptionV3 Test Accuracy: {inception_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
