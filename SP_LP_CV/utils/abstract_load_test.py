import numpy as np
import torch
import os
import faiss
from .utils import cal_recall

class LoadTest:
    def __init__(self, args, task_name, feature_dir):
        self.task_name = task_name
        self.feature_dir = feature_dir
    

    def load_features(self, feature1_path, feature2_path):
        if feature1_path.endswith(".npy"):
            feature1 = np.load(os.path.join(feature1_path))
        else:
            feature1 = torch.load(os.path.join(feature1_path), map_location="cpu")
            feature1 = feature1.numpy()
        
        if feature2_path.endswith(".npy"):
            feature2 = np.load(os.path.join(feature2_path))
        else:
            feature2 = torch.load(os.path.join(feature2_path), map_location="cpu")
            feature2 = feature2.numpy()

        feature1 = feature1.reshape(feature1.shape[0], -1)
        feature2 = feature2.reshape(feature2.shape[0], -1)
        return feature1, feature2
    
    def test(self, feature1, feature2, feature1_transformed):
        raise NotImplementedError
    
    def test_recall(self, feature1, feature2, feature1_transformed):
        pass

    @classmethod
    def test_recall(cls, feature1, feature2, feature1_transformed, only_1_index, only_2_index, search_range):
        query_one = feature1[:100]
        query_two = feature2[:100]

        index_feature1 = faiss.IndexFlatL2(feature1.shape[1])
        index_feature2 = faiss.IndexFlatL2(feature2.shape[1])
        index_feature1_transformed = faiss.IndexFlatL2(feature1_transformed.shape[1])

        index_feature1.add(feature1)
        index_feature2.add(feature2)
        index_feature1_transformed.add(feature1_transformed)

        recall_dict = {}
        for k in search_range:
            ans1 = index_feature1.search(query_one, k)
            ans2 = index_feature2.search(query_two, k)
            ans3 = index_feature1_transformed.search(query_two, k)
            recall_ans = {
                f'1->1/2->2@{k}': cal_recall(ans1, ans2),
                f'2->1\'/1->1@{k}': cal_recall(ans1, ans3),
                f'2->1\'/2->2@{k}': cal_recall(ans2, ans3),
            }
            recall_dict.update(recall_ans)
        return recall_dict