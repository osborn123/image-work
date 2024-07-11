import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from zfnet import ZFNet

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy

def topk_test(model, device, test_loader, k=5):
    model.eval()
    topk_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.topk(k, dim=1, largest=True, sorted=True)
            topk_correct += sum([target[i] in pred[i] for i in range(len(target))])

    topk_accuracy = 100. * topk_correct / len(test_loader.dataset)
    print(f'\nTest set: Top-{k} Accuracy: {topk_correct}/{len(test_loader.dataset)} ({topk_accuracy:.0f}%)\n')
    return topk_accuracy

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道图像
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 模型、损失函数和优化器
    model = ZFNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和测试
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    # 执行 top-k 测试
    topk_test(model, device, test_loader, k=5)

if __name__ == '__main__':
    main()
