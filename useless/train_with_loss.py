import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import gzip
import struct
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vgg19 = models.vgg19(init_weights=True).to(device)
inception_v3 = models.inception_v3(init_weights=True, aux_logits=False).to(device)


vgg19_weights = torch.load('/Users/liuxuchen/imagetesting/model_weights/vgg19-dcbb9e9d.pth')
inception_v3_weights = torch.load('/Users/liuxuchen/imagetesting/model_weights/inception_v3_google-1a9a5a14.pth')

vgg19.load_state_dict(vgg19_weights)

# 过滤掉InceptionV3的辅助逻辑权重
filtered_inception_weights = {k: v for k, v in inception_v3_weights.items() if not k.startswith('AuxLogits')}
inception_v3.load_state_dict(filtered_inception_weights)

#
vgg19_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inception_preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def infer_image_vgg19(image):
    img = Image.fromarray(image).convert('RGB')
    img = vgg19_preprocess(img)
    img.unsqueeze_(dim=0)
    output = vgg19(img.to(device))
    _, predicted = torch.max(output, 1)
    return predicted.item()

def infer_image_inception(image):
    img = Image.fromarray(image).convert('RGB')
    img = inception_preprocess(img)
    img.unsqueeze_(dim=0)
    output = inception_v3(img.to(device))
    _, predicted = torch.max(output, 1)
    return predicted.item()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
vgg19_optimizer = optim.SGD(vgg19.parameters(), lr=0.001, momentum=0.9)
inception_optimizer = optim.SGD(inception_v3.parameters(), lr=0.001, momentum=0.9)

def train_model(model, optimizer, images, labels, preprocess, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        img = Image.fromarray(image).convert('RGB')
        img = preprocess(img)
        img.unsqueeze_(dim=0)
        img = img.to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        if predicted == label:
            correct += 1
    
    accuracy = correct / len(images)
    avg_loss = total_loss / len(images)
    return avg_loss, accuracy

def main():
    images = read_mnist_images('/Users/liuxuchen/imagetesting/MNIST-master/train-images-idx3-ubyte.gz')
    labels = read_mnist_labels('/Users/liuxuchen/imagetesting/MNIST-master/train-labels-idx1-ubyte.gz')
    
    # 训练VGG19模型
    vgg19_loss, vgg19_accuracy = train_model(vgg19, vgg19_optimizer, images, labels, vgg19_preprocess, criterion, device)
    print(f'VGG19 Loss: {vgg19_loss:.4f}, Accuracy: {vgg19_accuracy * 100:.2f}%')
    
    # 训练InceptionV3模型
    inception_loss, inception_accuracy = train_model(inception_v3, inception_optimizer, images, labels, inception_preprocess, criterion, device)
    print(f'InceptionV3 Loss: {inception_loss:.4f}, Accuracy: {inception_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
