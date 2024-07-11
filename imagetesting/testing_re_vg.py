import sys
import os
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from keras.utils import to_categorical
from keras.optimizers import Adam

# 将模型目录添加到 Python 路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from vgg_16 import vgg_fm  # 调整导入路径
from resnet import resnet18  # 调整导入路径

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
test_images = load_mnist_images(r'C:\Users\liu18\image-work\imagetesting\MNIST-master\t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels(r'C:\Users\liu18\image-work\imagetesting\MNIST-master\t10k-labels-idx1-ubyte.gz')

# PyTorch 模型评估
def evaluate_pytorch_model(model, test_images, test_labels):
    model.eval()
    test_images_tensor = torch.tensor(test_images).permute(0, 3, 1, 2)  # reshape to (N, 1, 28, 28)
    test_labels_tensor = torch.tensor(test_labels)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    testloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"PyTorch Model Accuracy: {100 * correct / total}%")

# Keras 模型评估
def evaluate_keras_model(model, test_images, test_labels):
    test_images = test_images.astype('float32') / 255.0
    test_labels_one_hot = to_categorical(test_labels, 10)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(test_images, test_labels_one_hot)
    print(f"Keras Model Accuracy: {accuracy * 100}%")

# 加载并评估 PyTorch 模型
pytorch_model = resnet18(num_classes=10)  # MNIST 有 10 个类别
evaluate_pytorch_model(pytorch_model, test_images, test_labels)

# 加载并评估 Keras 模型
keras_model = vgg_fm((28, 28, 1))
evaluate_keras_model(keras_model, test_images, test_labels)
