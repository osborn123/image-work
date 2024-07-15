from .abstract_load_test import LoadTest
from generate_feature.test_lp import train_model, load_data, test_lp, decode
import torch
import numpy as np
import os

class LinkPredictionLoadTest(LoadTest):
    def __init__(self, args, task_name, feature_dir):
        super().__init__(args, task_name, feature_dir)
        self.model_name1, self.data1 = args.feature1_path.replace(".pt", "").split("_")
        self.model_name2, self.data2 = args.feature2_path.replace(".pt", "").split("_")
    
    def load_features(self, *args, **kwargs):
        feature1, feature2 = self.load_two_features(*args, **kwargs)
        self.feature1 = feature1
        self.feature2 = feature2
        return feature1, feature2

    def test(self, *args, **kwargs):
        return self.test_lp(*args, **kwargs)
    
    def test_lp(self, args, feature1, feature2, feature1_transformed, only_1_index, only_2_index, overlap_index, verbose=False):
        if self.data1 == self.data2:
            data = load_data(self.data1, device="cpu")
        else:
            raise ValueError("data not match")
        
        # _, model1 = train_model(encoder_name=self.model_name1, verbose=verbose)
        _, model2 = train_model(encoder_name=self.model_name2, verbose=verbose)

        # new_feature = feature1.copy()
        # new_feature = feature1_transformed[only_1_index]
        new_feature = feature1_transformed
        # auc_ori, ap_ori = model2.test(torch.from_numpy(feature2), data.test_pos_edge_index, data.test_neg_edge_index)
        # auc, ap = model2.test(torch.from_numpy(new_feature), data.test_pos_edge_index, data.test_neg_edge_index)
        auc_ori, ap_ori = test_lp(torch.from_numpy(feature2), data.test_pos_edge_index, data.test_neg_edge_index, torch.from_numpy(feature2))
        auc, ap = test_lp(torch.from_numpy(new_feature), data.test_pos_edge_index, data.test_neg_edge_index, torch.from_numpy(feature2))
        
        return {"auc_ori": auc_ori, 
                "ap_ori": ap_ori, 
                "auc_tranformed": auc, 
                "ap_transformed": ap}

    def load_two_features(self, args):
        # load feature
        args.saved_features_dir += "/link_prediction"
        if args.feature1_path.endswith(".npy"):
            feature1 = np.load(os.path.join(args.saved_features_dir, args.feature1_path))
        else:
            feature1 = torch.load(os.path.join(args.saved_features_dir, args.feature1_path), map_location="cpu")
            feature1 = feature1.numpy()
        
        if args.feature2_path.endswith(".npy"):
            feature2 = np.load(os.path.join(args.saved_features_dir, args.feature2_path))
        else:
            feature2 = torch.load(os.path.join(args.saved_features_dir, args.feature2_path), map_location="cpu")
            feature2 = feature2.numpy()

        feature1 = feature1.reshape(feature1.shape[0], -1)
        feature2 = feature2.reshape(feature2.shape[0], -1)
        return feature1, feature2