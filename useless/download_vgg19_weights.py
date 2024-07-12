import os
import torch
import torchvision.models as models

# 指定模型权重文件的URL
model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

# 指定下载路径
download_dir = './model_weights'
os.makedirs(download_dir, exist_ok=True)
download_path = os.path.join(download_dir, 'vgg19-dcbb9e9d.pth')

# 下载模型权重文件
if not os.path.exists(download_path):
    torch.hub.download_url_to_file(model_url, download_path)

# 加载VGG19模型，但不加载预训练权重
model = models.vgg19(weights = True)

# 加载下载的权重文件到模型中
model.load_state_dict(torch.load(download_path))

# 切换模型到评估模式
model.eval()

# 打印模型结构（可选）
print(model)
