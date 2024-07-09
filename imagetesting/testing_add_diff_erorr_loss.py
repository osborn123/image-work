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
import matplotlib.pyplot as plt

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
inception_v3 = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True).to(device)

# 加载权重
vgg19_weights = torch.load('imagetesting/model_weights/vgg19-dcbb9e9d.pth')
inception_v3_weights = torch.load('imagetesting/model_weights/inception_v3_google-1a9a5a14.pth')

# 加载权重到模型
vgg19.load_state_dict(vgg19_weights)
inception_v3.load_state_dict(inception_v3_weights)

# 移除 InceptionV3 模型的辅助分类器
inception_v3.aux_logits = False
inception_v3.AuxLogits = None

# 自定义MNIST数据集类
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image).convert('RGB')
        image = transforms.functional.resize(image, (299, 299))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def calculate_errors(predictions, labels):
    predictions = predictions.float()
    labels = labels.float()
    abs_errors = torch.abs(predictions - labels)
    max_abs_error = torch.max(abs_errors).item()
    mean_abs_error = torch.mean(abs_errors).item()
    mse = torch.mean((predictions - labels) ** 2).item()
    return max_abs_error, mean_abs_error, mse

def train_model(model, optimizer, dataloader, criterion, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        max_abs_errors = []
        mean_abs_errors = []
        mses = []
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
            
            # Calculate and store errors
            max_abs_error, mean_abs_error, mse = calculate_errors(predicted, labels)
            max_abs_errors.append(max_abs_error)
            mean_abs_errors.append(mean_abs_error)
            mses.append(mse)
        
        accuracy = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader)
        avg_max_abs_error = np.mean(max_abs_errors)
        avg_mean_abs_error = np.mean(mean_abs_errors)
        avg_mse = np.mean(mses)
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Max Abs Error: {avg_max_abs_error:.4f}, Mean Abs Error: {avg_mean_abs_error:.4f}, MSE: {avg_mse:.4f}')

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    max_abs_errors = []
    mean_abs_errors = []
    mses = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            
            # Calculate and store errors
            max_abs_error, mean_abs_error, mse = calculate_errors(predicted, labels)
            max_abs_errors.append(max_abs_error)
            mean_abs_errors.append(mean_abs_error)
            mses.append(mse)
    
    accuracy = correct / len(dataloader.dataset)
    avg_max_abs_error = np.mean(max_abs_errors)
    avg_mean_abs_error = np.mean(mean_abs_errors)
    avg_mse = np.mean(mses)
    
    return accuracy, avg_max_abs_error, avg_mean_abs_error, avg_mse

def plot_errors(error_dict):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for i, (error_type, values) in enumerate(error_dict.items()):
        for model_name, errors in values.items():
            axes[i].plot(errors, label=model_name)
        axes[i].set_title(error_type)
        axes[i].legend()
    plt.tight_layout()
    plt.show()

def main():
    # 读取训练集和测试集
    train_images = read_mnist_images('imagetesting/MNIST-master/train-images-idx3-ubyte.gz')
    train_labels = read_mnist_labels('imagetesting/MNIST-master/train-labels-idx1-ubyte.gz')
    test_images = read_mnist_images('imagetesting/MNIST-master/t10k-images-idx3-ubyte.gz')
    test_labels = read_mnist_labels('imagetesting/MNIST-master/t10k-labels-idx1-ubyte.gz')
    
    # 创建数据集和数据加载器
    original_train_dataset = MNISTDataset(train_images, train_labels)
    original_test_dataset = MNISTDataset(test_images, test_labels)
    
    original_train_dataloader = DataLoader(original_train_dataset, batch_size=32, shuffle=True)
    original_test_dataloader = DataLoader(original_test_dataset, batch_size=32, shuffle=False)

    # 定义损失函数和优化器
    criterion_list = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "MSELoss": nn.MSELoss(),
    }
    vgg19_optimizer = optim.SGD(vgg19.parameters(), lr=0.001, momentum=0.9)
    inception_optimizer = optim.SGD(inception_v3.parameters(), lr=0.001, momentum=0.9)

    # 存储误差
    error_dict = {
        "Max Absolute Error": {},
        "Mean Absolute Error": {},
        "Mean Square Error": {}
    }

    # 训练和测试模型
    for criterion_name, criterion in criterion_list.items():
        for model_name, model, optimizer, train_dataloader, test_dataloader in [
            ("VGG19", vgg19, vgg19_optimizer, original_train_dataloader, original_test_dataloader),
            ("InceptionV3", inception_v3, inception_optimizer, original_train_dataloader, original_test_dataloader),
        ]:
            print(f"Training {model_name} model with {criterion_name} loss...")
            train_model(model, optimizer, train_dataloader, criterion, device, epochs=5)
            
            print(f"Testing {model_name} model with {criterion_name} loss...")
            accuracy, max_abs_error, mean_abs_error, mse = test_model(model, test_dataloader, device)
            print(f'{model_name} Test Accuracy with {criterion_name} loss: {accuracy * 100:.2f}%, Max Abs Error: {max_abs_error:.4f}, Mean Abs Error: {mean_abs_error:.4f}, MSE: {mse:.4f}')

            # 记录误差
            if model_name not in error_dict["Max Absolute Error"]:
                error_dict["Max Absolute Error"][model_name] = []
                error_dict["Mean Absolute Error"][model_name] = []
                error_dict["Mean Square Error"][model_name] = []
            
            error_dict["Max Absolute Error"][model_name].append(max_abs_error)
            error_dict["Mean Absolute Error"][model_name].append(mean_abs_error)
            error_dict["Mean Square Error"][model_name].append(mse)

    plot_errors(error_dict)

if __name__ == "__main__":
    main()
