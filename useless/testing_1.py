import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import gzip
import struct

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
vgg19 = models.vgg19(init_weights=True).to(device)
inception_v3 = models.inception_v3(init_weights=True, aux_logits=False).to(device)

# Load weights
vgg19_weights = torch.load('/Users/liuxuchen/imagetesting/model_weights/vgg19-dcbb9e9d.pth')
inception_v3_weights = torch.load('/Users/liuxuchen/imagetesting/model_weights/inception_v3_google-1a9a5a14.pth')

# Load weights into models
vgg19.load_state_dict(vgg19_weights)

# Filter out auxiliary logits weights for InceptionV3
filtered_inception_weights = {k: v for k, v in inception_v3_weights.items() if not k.startswith('AuxLogits')}
inception_v3.load_state_dict(filtered_inception_weights)

# Define preprocessing steps
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

def test_accuracy(model_name):
    images = read_mnist_images('/Users/liuxuchen/imagetesting/MNIST-master/t10k-images-idx3-ubyte.gz')
    labels = read_mnist_labels('/Users/liuxuchen/imagetesting/MNIST-master/t10k-labels-idx1-ubyte.gz')

    correct = 0
    total = len(images)

    for i in range(total):
        image = images[i]
        label = labels[i]
        
        if model_name == 'vgg19':
            predicted_label = infer_image_vgg19(image)
        elif model_name == 'inception':
            predicted_label = infer_image_inception(image)
        
        if predicted_label == label:
            correct += 1

    accuracy = correct / total
    print(f'{model_name.upper()} Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    test_accuracy('vgg19')
    test_accuracy('inception')
