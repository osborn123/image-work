import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import faiss
import os

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(224),  # 调整图像大小以适应VGG的输入
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将灰度图像转换为彩色图像
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])

# 加载MNIST数据集
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)

# 将模型移到GPU上（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用预训练的VGG模型
model = models.vgg16(pretrained=True).to(device)

# 替换最后的全连接层以适应MNIST的类别数（10类数字）
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 10).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
num_epochs = 1


# # 训练模型
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in tqdm(train_loader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# 添加瓶颈层
model.eval()  # 设置为评估模式
num_ftrs = model.classifier[6].in_features
bottleneck_features = 1000  # 您可以根据需要调整这个值
model.classifier[6] = nn.Linear(num_ftrs, bottleneck_features)
model.classifier.append(nn.Linear(bottleneck_features, 10))  # 添加新的全连接层

model.to(device)
# 冻结VGG模型的参数
for param in model.parameters():
    param.requires_grad = False

# 解冻瓶颈层和新添加的全连接层的参数
for param in model.classifier[-2:].parameters():
    param.requires_grad = True

# 重新定义优化器，只优化瓶颈层和新添加的全连接层的参数
optimizer = optim.SGD(model.classifier[-2:].parameters(), lr=1e-3, momentum=0.9)

# 再次训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
# print test acc
# total_correct = 0
# for images, labels in test_loader:
#     images, labels = images.to(device), labels.to(device)
#     outputs = model(images)
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == labels).sum().item()
#     total_correct += correct
# print(f"Test Accuracy: {total_correct/len(test_dataset):.4f}")
# 获取MNIST表征
saved_features = []
with torch.no_grad():
    model.eval()
    for images, _ in test_loader:
        images = images.to(device)
        features = model.features(images)
        features = features.view(features.size(0), -1)  # 展平特征
        features = model.classifier[:7](features)  # 通过瓶颈层
        saved_features.append(features)
saved_features = torch.cat(saved_features, 0)
print("Features shape:", saved_features.shape)

# 保存特征
if not os.path.exists("./saved_features"):
    os.makedirs("./saved_features")
torch.save(saved_features, f"./saved_features/vgg_mnist_reduced_features_{num_epochs}.pt")

# make t-sne figure
import numpy as np
import matplotlib.pyplot as plt
x = saved_features.cpu().numpy()
y = train_dataset.targets.numpy()
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_2d = tsne.fit_transform(x)
target_ids = range(len(np.unique(y)))
plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
for i, c, label in zip(target_ids, colors, np.unique(y)):
    plt.scatter(x_2d[y == i, 0], x_2d[y == i, 1], c=c, label=label)
    break
plt.legend()
plt.savefig("tsne.png")
plt.clf()


import numpy as np
query_index = np.random.choice(np.arange(len(saved_features)), 10)
query_feature = saved_features[query_index].cpu().numpy()

index_flat = faiss.IndexFlatL2(saved_features.shape[1])

index_flat.add(saved_features.cpu().numpy())
ans = index_flat.search(query_feature, 10)

label = train_dataset.targets.numpy()
print(f"【Target】", label[query_index])
for i in range(len(ans[1])):
    print(f"【Top 10】", label[ans[1][i]])



