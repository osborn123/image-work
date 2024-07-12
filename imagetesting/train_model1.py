import argparse
import gzip
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from resnet import resnet18

def load_mnist_images(filename):
    with gzip.open(filename, 'r') as f:
        f.read(16)
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 28, 28, 1)
        data = data / 127.5 - 1
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'r') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
        return labels

def compute_mean_std(images):
    mean = np.mean(images)
    std = np.std(images)
    return mean, std

def train_pytorch_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Time: {epoch_time:.2f} seconds")

def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"PyTorch Model Accuracy: {100 * correct / total}%")
    return 100 * correct / total

def topk_test(model, device, test_loader, k=5):
    model.eval()
    model.to(device)
    topk_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
            topk_correct += sum([labels[i] in pred[i] for i in range(len(labels))])
    topk_accuracy = 100. * topk_correct / len(test_loader.dataset)
    print(f'\nTest set: Top-{k} Accuracy: {topk_correct}/{len(test_loader.dataset)} ({topk_accuracy:.0f}%)\n')
    return topk_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5, help='Top-k value for accuracy evaluation.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model.')
    args = parser.parse_args()

    print("CUDA Available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count: ", torch.cuda.device_count())
        print("Current device: ", torch.cuda.current_device())
        print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    train_images = load_mnist_images(os.path.join('data', 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join('data', 'train-labels-idx1-ubyte.gz'))
    test_images = load_mnist_images(os.path.join('data', 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join('data', 't10k-labels-idx1-ubyte.gz'))

    mean, std = compute_mean_std(train_images)

    transform = transforms.Compose([
        transforms.Normalize((mean,), (std,))
    ])

    train_images_tensor = transform(torch.tensor(train_images).permute(0, 3, 1, 2))
    train_labels_tensor = torch.tensor(train_labels)
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_images_tensor = transform(torch.tensor(test_images).permute(0, 3, 1, 2))
    test_labels_tensor = torch.tensor(test_labels)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    overlap_rates = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for rate in overlap_rates:
        print(f"\nTraining and testing with overlap rate: {int(rate * 100)}%")
        train_pytorch_model(model, train_loader, criterion, optimizer, device, epochs=args.epochs)
        accuracy = evaluate_pytorch_model(model, test_loader, device)
        topk_accuracy = topk_test(model, device, test_loader, k=args.top_k)
        results.append({
            'overlap_rate': rate,
            'accuracy': accuracy,
            'topk_accuracy': topk_accuracy
        })

    for result in results:
        print(f"Overlap Rate: {int(result['overlap_rate'] * 100)}%")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Top-k Accuracy: {result['topk_accuracy']}\n")

if __name__ == '__main__':
    main()
