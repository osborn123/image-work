from .abstract_load_test import LoadTest
import torch
import numpy as np
import faiss
from torchvision import datasets, transforms
from generate_feature.test_image_classification import transform, train_dataset, test_dataset, train_loader, test_loader, vggmodel, vggnewmodel


class ImageClassificationLoadTest(LoadTest):
    def __init__(self, args, task_name=None, feature_dir=None):
        if task_name is None:
            task_name = "image_classification"
        if feature_dir is None:
            feature_dir = "./saved_features/image_classification"
        else:
            feature_dir = f"{feature_dir}/{task_name}"
        super().__init__(args, task_name, feature_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # self.test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=64, shuffle=True)
        # self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=64, shuffle=False)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model1 = vggmodel 
        # self.model2 = zfmodel
        self.model2 = vggnewmodel
    
    def load_features(self, args, feature1_name=None, feature2_name=None):
        if feature1_name is None:
            feature1_name = args.feature1_path
        if feature2_name is None:
            feature2_name = args.feature2_path
        # feature1, feature2 = self.load_two_features(feature1_name, feature2_name)
        feature1_path = f"{self.feature_dir}/{feature1_name}"
        feature2_path = f"{self.feature_dir}/{feature2_name}"
        feature1, feature2 = super().load_features(feature1_path, feature2_path)
        return feature1, feature2

    # def load_two_features(self, feature1_name, feature2_name):
    #     feature1 = torch.load(f"{self.feature_dir}/{feature1_name}")
    #     feature2 = torch.load(f"{self.feature_dir}/{feature2_name}")
    #     return feature1, feature2
    
    def test(self, args, feature1, feature2, feature1_transformed=None, only_1_index=False, only_2_index=False, overlap_index=None):
        return self.test_topk_accuracy(feature1, feature2, feature1_transformed)
        # return {}

    def test_topk_accuracy(self, feature1, feature2, feature1_transformed):
        device = self.model1.fc.weight.device
        if isinstance(feature1, np.ndarray):
            feature1 = torch.from_numpy(feature1)
        if isinstance(feature2, np.ndarray):
            feature2 = torch.from_numpy(feature2)
        if isinstance(feature1_transformed, np.ndarray):
            feature1_transformed = torch.from_numpy(feature1_transformed)
        feature1, feature2, feature1_transformed = feature1.to(device), feature2.to(device), \
            feature1_transformed.to(device)

        self.model1.fc(feature1)
        from sklearn.metrics import accuracy_score
        acc1 = accuracy_score(self.test_dataset.targets, self.model1.fc(feature1).argmax(dim=1).cpu().numpy())
        acc2 = accuracy_score(self.test_dataset.targets, self.model2.fc(feature2).argmax(dim=1).cpu().numpy())
        acc1_transformed = accuracy_score(self.test_dataset.targets, self.model2.fc(feature1_transformed).argmax(dim=1).cpu().numpy())
        return {
            "acc1": acc1,
            "acc2": acc2,
            "acc1_transformed": acc1_transformed
        }
        # test accuracy on feature1 & model1
        # test accuracy on feature2 & model2
        # test accuracy on feature1_transofrmed & model2
    
    # def test_topk_accuracy(self, feature1, feature2, feature1_transformed):
    #     topk_acc1 = self.topk_test(feature1, self.test_dataset.test_labels)
    #     topk_acc2 = self.topk_test(feature2, self.test_dataset.test_labels)
    #     topk_acc1_transformed = self.topk_test(feature1_transformed, self.test_dataset.test_labels)
    #     return {
    #         "topk_acc1": topk_acc1,
    #         "topk_acc2": topk_acc2,
    #         "topk_acc1_transformed": topk_acc1_transformed
    #     }

    # def topk_test(self, features, labels, k=5):
    #     total = 0
    #     topk_correct = 0

    #     index = faiss.IndexFlatL2(features.shape[1])
    #     index.add(features)
    #     D, I = index.search(features, k)

    #     for i in range(features.shape[0]):
    #         if labels[i] in labels[I[i]]:
    #             topk_correct += 1
    #         total += 1

    #     topk_accuracy = 100. * topk_correct / total
    #     print(f'\nTest set: Top-{k} Accuracy: {topk_correct}/{total} ({topk_accuracy:.0f}%)\n')
    #     return topk_accuracy
