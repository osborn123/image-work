import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(299),  # 调整图像大小以适应InceptionV3的输入
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将灰度图像转换为彩色图像
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])


# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# use train dataset as test dataset
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print("Use train dataset as test dataset")
test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)


# 使用预训练的InceptionV3模型
model = models.inception_v3(pretrained=True)

# 替换最后的全连接层以适应MNIST的类别数（10类数字）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 将模型移到GPU上（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

import os
if os.path.exists("inception_mnist.pth"):
    model.load_state_dict(torch.load("inception_mnist.pth"))
    print("Loaded model from disk")
else:
    # 训练模型
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, features = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "inception_mnist.pth")
    print("Saved model to disk")

# 在测试集上评估模型
# model.eval()
# total = 0
# correct = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy of the model on the test images: {100 * correct / total}%")

# 获取MNIST表征
# 关闭梯度跟踪
# with torch.no_grad():
#     model.eval()
#     # 选择一个批次的数据
#     for images, _ in test_loader:
#         images = images.to(device)
#         # 获取特征
#         features = model.Conv2d_1a_3x3(images)
#         features = model.Conv2d_2a_3x3(features)
#         features = model.Conv2d_2b_3x3(features)
#         features = model.maxpool1(features)
#         features = model.Conv2d_3b_1x1(features)
#         features = model.Conv2d_4a_3x3(features)
#         features = model.maxpool2(features)
#         features = model.Mixed_5b(features)
#         features = model.Mixed_5c(features)
#         features = model.Mixed_5d(features)
#         features = model.Mixed_6a(features)
#         features = model.Mixed_6b(features)
#         features = model.Mixed_6c(features)
#         features = model.Mixed_6d(features)
#         features = model.Mixed_6e(features)
#         features = model.Mixed_7a(features)
#         features = model.Mixed_7b(features)
#         features = model.Mixed_7c(features)
#         # 打印特征维度
#         print(features.shape)
        # 如果需要，可以将特征保存到文件中或进行进一步处理
        # break  # 只获取一个批次的数据作为示例
saved_features = []
with torch.no_grad():
    model.train()
    # 选择一个批次的数据
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        # 获取特征
        out, features = model(images)
        saved_features.append(features)
saved_features = torch.cat(saved_features, 0)
torch.save(saved_features, "./saved_features/inception_mnist_features.pt")
