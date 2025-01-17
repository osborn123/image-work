import argparse
import numpy as np
import faiss
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import ResNet
from zfnet import ZFNet
from linear_transform import LinearTransform
from mmd_transform import MMDTransform
import wandb

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

def test_recall(feature1, feature2, feature2_selected, only_1_index, only_2_index, search_range):
    index_feature1 = faiss.IndexFlatL2(feature1.shape[1])
    index_feature2 = faiss.IndexFlatL2(feature2.shape[1])

    index_feature1.add(feature1[only_2_index])
    index_feature2.add(feature2_selected[only_2_index])

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

def topk_test(features, labels, k=5):
    total = 0
    topk_correct = 0

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    D, I = index.search(features, k)

    for i in range(features.shape[0]):
        if labels[i] in labels[I[i]]:
            topk_correct += 1
        total += 1

    topk_accuracy = 100. * topk_correct / total
    print(f'\nTest set: Top-{k} Accuracy: {topk_correct}/{total} ({topk_accuracy:.0f}%)\n')
    return topk_accuracy

def main(args):
    seed_everything(42)
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale image to 3-channel image
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=250, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = ResNet(num_classes=10)
    zfnet_model = ZFNet(num_classes=10)

    print("Extracting features using ResNet model...")
    feature1 = extract_features(resnet_model, test_loader, device)
    print("Extracting features using ZFNet model...")
    feature2 = extract_features(zfnet_model, test_loader, device)

    indices = np.arange(feature1.shape[0])  
    np.random.shuffle(indices)
    
    overlap_rate = args.overlap_rate
    print(f"\nTesting with overlap rate: {int(overlap_rate * 100)}%")
    overlap_count = int(len(indices) * overlap_rate)
    split = int(len(indices) * 0.8)
    only_1_index = indices[:split - overlap_count]
    only_2_index = indices[split - overlap_count:]
    overlap_index = only_2_index[:overlap_count]

    if args.transform_method == "linear":
        print("Performing linear transform...")
        feature2_selected = LinearTransform(args, feature2, feature1)
    elif args.transform_method == "class_linear":
        feature2_selected = feature2.copy()
    elif args.transform_method == "" or args.transform_method == "none":
        feature2_selected = feature2
    elif args.transform_method == "MMD":
        print("Performing MMD transform...")
        feature1_overlapd = feature1[overlap_index]
        np.random.shuffle(overlap_index)
        feature2_overlapd = feature2[overlap_index]
        feature1, feature2_selected = MMDTransform(args, feature2_overlapd, feature1_overlapd, transformed_feature1=feature1, transformed_feature2=feature2)
    else:
        raise ValueError("transform method not found")

    print("Evaluating recall...")
    recall_results, mean_recall, max_recall, mean_square_recall = test_recall(feature1, feature2, feature2_selected, only_1_index, only_2_index, args.search_range)
    print("Recall Results:", recall_results)
    print("Mean Recall:", mean_recall)
    print("Max Recall:", max_recall)
    print("Mean Square Recall:", mean_square_recall)

    print("Evaluating Top-k accuracy...")
    test_labels = np.array([label for _, label in test_dataset])
    topk_accuracy = topk_test(feature2_selected, test_labels, k=args.top_k)
    print("Top-k Accuracy:", topk_accuracy)

    print(f"Final Results for overlap rate {int(overlap_rate * 100)}%:")
    print(f"Recall Results: {recall_results}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Max Recall: {max_recall}")
    print(f"Mean Square Recall: {mean_square_recall}")
    print(f"Top-k Accuracy: {topk_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_range", type=int, nargs='+', default=[10, 20, 50, 100], help="the search range for faiss search")
    parser.add_argument("--top_k", type=int, default=5, help="the top-k value for top-k accuracy")
    parser.add_argument("--transform_method", type=str, default="none", help="feature transform method: none, linear, class_linear, or MMD")
    parser.add_argument("--transform_lr", type=float, default=1e-4, help="learning rate for feature transform")
    parser.add_argument("--transform_epoch", type=int, default=1000, help="number of epochs for feature transform")
    parser.add_argument("--transform_batch_size", type=int, default=256, help="batch size for feature transform")
    parser.add_argument("--device", type=int, default=0, help="device id for training")
    parser.add_argument("--overlap_rate", type=float, default=0, help="overlap rate for the experiment")
    parser.add_argument("--overlap_presets", type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="preset overlap rates for the experiment")

    args = parser.parse_args()

    # Initialize wandb outside the loop
    wandb.init(project="feature-transform", config=args, mode="disabled")

    # Loop over overlap presets if specified
    for overlap_rate in args.overlap_presets:
        args.overlap_rate = overlap_rate
        wandb.run.name = f"overlap_rate_{overlap_rate}"
        main(args)

    wandb.finish()
