import gzip
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import urllib.request

# 解压文件并读取图像数据
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# 解压文件并读取标签数据
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# 加载测试集图像和标签数据
test_images = load_mnist_images('/Users/liuxuchen/imagetesting/MNIST-master/t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('/Users/liuxuchen/imagetesting/MNIST-master/t10k-labels-idx1-ubyte.gz')

# 下载ImageNet类别映射文件
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = urllib.request.urlopen(url)
class_names = [line.strip() for line in response]

# 加载VGG19模型并使用预训练权重
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
model.eval()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),  # VGG19 使用 256x256 输入
    transforms.CenterCrop(224),  # VGG19 使用 224x224 中心裁剪
    transforms.Grayscale(num_output_channels=3),  # MNIST是灰度图像，VGG19需要RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 处理并推理整个测试集的图像
def evaluate_accuracy(images, labels, model, device):
    model.eval()
    correct = 0
    total = 0
    for img, label in zip(images, labels):
        img = Image.fromarray(img, mode='L')  # 将numpy数组转换为PIL图像
        img = preprocess(img)  # 预处理图像
        img = img.unsqueeze(0).to(device)  # 添加batch维度并移动到GPU（如果可用）
        
        with torch.no_grad():
            output = model(img)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted = torch.max(probs, 0)
        
        # 使用真实标签计算准确性
        if predicted.item() == label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

# 评估整个测试集的准确性
accuracy = evaluate_accuracy(test_images, test_labels, model, device)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
