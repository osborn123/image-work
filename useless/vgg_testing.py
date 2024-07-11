import gzip
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 解压文件并读取图像数据
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# 加载测试集图像数据
test_images = load_mnist_images('/Users/liuxuchen/imagetesting/MNIST-master/t10k-images-idx3-ubyte.gz')

# 加载VGG19模型并使用预训练权重
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
model.eval()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # MNIST是灰度图像，VGG需要RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 处理并推理一个批次的图像
def process_and_infer(images, model, device):
    model.eval()
    results = []
    for img in images:
        img = Image.fromarray(img, mode='L')  # 将numpy数组转换为PIL图像
        img = preprocess(img)  # 预处理图像
        img = img.unsqueeze(0).to(device)  # 添加batch维度并移动到GPU（如果可用）
        
        with torch.no_grad():
            output = model(img)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probs, 5)
        
        results.append((top5_prob.cpu().numpy(), top5_catid.cpu().numpy()))
    
    return results

# 选择前5个图像进行推理
results = process_and_infer(test_images[:5], model, device)

# 打印推理结果
for i, (probs, catids) in enumerate(results):
    print(f"Image {i+1}:")
    for prob, catid in zip(probs, catids):
        print(f"  Category ID: {catid}, Probability: {prob}")
