import argparse
import numpy as np
import faiss, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import ResNet
from zfnet import ZFNet

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def extract_features(model, dataloader, device):
    model.eval()
    model.to(device)
    features = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            print(f'Batch {i+1}/{len(dataloader)}: Extracted features shape: {outputs.shape}')
    return np.concatenate(features, axis=0)

def test_recall(feature1, feature2, only_2_index, search_range):
    index_feature1 = faiss.IndexFlatL2(feature1.shape[1])
    index_feature2 = faiss.IndexFlatL2(feature2.shape[1])

    index_feature1.add(feature1[only_2_index])
    index_feature2.add(feature2[only_2_index])

    recall_list = []
    for k in search_range:
        ans1 = index_feature1.search(feature1[:1000], k)
        ans2 = index_feature2.search(feature1[:1000], k)
        recall = (ans1[1] == ans2[1]).mean()
        recall_list.append(recall)
        print(f"Recall@{k}: {recall}")
    mean_recall = np.mean(recall_list)
    max_recall = np.max(recall_list)
    mean_square_recall = np.mean(np.square(recall_list))
    print(f'Mean Recall: {mean_recall}, Max Recall: {max_recall}, Mean Square Recall: {mean_square_recall}')
    return recall_list, mean_recall, max_recall, mean_square_recall

def topk_test(model, device, test_loader, k=5):
    model.eval()
    model.to(device)
    topk_correct = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
            topk_correct += sum([labels[i] in pred[i] for i in range(len(labels))])
            print(f'Batch {i+1}/{len(test_loader)}: Top-{k} accuracy for this batch: {topk_correct / ((i+1) * test_loader.batch_size):.4f}')

    topk_accuracy = 100. * topk_correct / len(test_loader.dataset)
    print(f'\nTest set: Top-{k} Accuracy: {topk_correct}/{len(test_loader.dataset)} ({topk_accuracy:.0f}%)\n')
    return topk_accuracy

def main(args):
    seed_everything(42)
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为三通道图像
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=250, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == "resnet":
        model = ResNet(num_classes=10)
    elif args.model == "zfnet":
        model = ZFNet(num_classes=10)
    else:
        raise ValueError("Unsupported model type. Choose either 'resnet' or 'zfnet'.")

    print(f"Extracting features using {args.model} model...")
    feature1 = extract_features(model, test_loader, device)

    indices = np.arange(feature1.shape[0])
    np.random.shuffle(indices)
    
    overlap_rates = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    search_range = [10, 20, 50, 100]
    results = []

    for overlap_rate in overlap_rates:
        print(f"\nTesting with overlap rate: {int(overlap_rate * 100)}%")
        overlap_count = int(len(indices) * overlap_rate)
        split = int(len(indices) * 0.8)
        only_1_index = indices[:split - overlap_count]
        only_2_index = indices[split - overlap_count:]
        overlap_index = only_2_index[:overlap_count]

        print("Evaluating recall...")
        recall_results, mean_recall, max_recall, mean_square_recall = test_recall(feature1, feature1, only_2_index, search_range)
        print("Recall Results:", recall_results)
        print("Mean Recall:", mean_recall)
        print("Max Recall:", max_recall)
        print("Mean Square Recall:", mean_square_recall)

        print("Evaluating Top-k accuracy...")
        topk_accuracy = topk_test(model, device, test_loader, k=args.top_k)
        print("Top-k Accuracy:", topk_accuracy)

        results.append({
            'overlap_rate': overlap_rate,
            'recall_results': recall_results,
            'mean_recall': mean_recall,
            'max_recall': max_recall,
            'mean_square_recall': mean_square_recall,
            'topk_accuracy': topk_accuracy
        })
    
    print("\nFinal Results:")
    for result in results:
        print(f"Overlap Rate: {int(result['overlap_rate'] * 100)}%")
        print(f"Recall Results: {result['recall_results']}")
        print(f"Mean Recall: {result['mean_recall']}")
        print(f"Max Recall: {result['max_recall']}")
        print(f"Mean Square Recall: {result['mean_square_recall']}")
        print(f"Top-k Accuracy: {result['topk_accuracy']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model type: resnet or zfnet")
    parser.add_argument("--search_range", type=int, nargs='+', default=[10, 20, 50, 100], help="the search range for faiss search")
    parser.add_argument("--top_k", type=int, default=5, help="the top-k value for top-k accuracy")
    parser.add_argument("--device", type=int, default=0, help="device id for training")
    args = parser.parse_args()
    main(args)
