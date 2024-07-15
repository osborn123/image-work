import numpy as np
import os
import torch
import torch.nn.functional as F
from .abstract_load_test import LoadTest

def calculate_errors(dist, max_value, dist_normalized, prefix):
    # 计算AE
    ae = torch.abs(dist_normalized - dist)

    # 计算MAE
    mae = ae.mean()

    # 计算ME
    me = ae.max()

    # 计算MRE
    # mre = (ae / dist).mean() if dist.sum() != 0 else torch.tensor(0.0)
    mask = dist != 0
    mre = (torch.masked_select(ae, mask) / torch.masked_select(dist, mask)).mean()


    # 计算MSE
    mse = torch.nn.functional.mse_loss(dist_normalized, dist)

    # 计算RMSE
    rmse = torch.sqrt(mse)

    return {
        f"【AE】{prefix}": np.round(ae.mean().item(), 4), 
        f"【MAE】{prefix}": np.round(mae.item(), 4), 
        f"【ME】{prefix}": np.round(me.item(), 4), 
        f"【MRE】{prefix}": np.round(mre.item(), 4), 
        f"【MSE】{prefix}": np.round(mse.item(), 4), 
        f"【RMSE】{prefix}": np.round(rmse.item(), 4)
    }

def compute_ae_loss(z, sp_dist, normalize="max"):
    # sp_dist = data.sp_dist_raw
    if normalize == "max":
        dist = torch.cdist(z, z, p=2)
        dist = dist / dist.max()
        dist *= sp_dist.max()
    ae_loss = torch.abs(sp_dist - dist).mean()
    return ae_loss

def compute_mse_loss(z, sp_dist):
    return F.mse_loss(z/z.max(), sp_dist/sp_dist.max())

def cal_ae_loss(dist, target_dist):
    ae = torch.abs(dist - target_dist)
    return ae.mean()

def cal_me_loss(dist, target_dist):
    ae = torch.abs(dist - target_dist)
    return ae.max()

class ShortestPathLoadTest(LoadTest):
    def __init__(self, args, task_name, feature_dir):
        super().__init__(args, task_name, feature_dir)
    
    def load_features(self, feature1_name, feature2_name):
        feature1, feature2, dist = self.load_two_features_and_dist(self.args)
        self.feature1 = feature1
        self.feature2 = feature2
        self.dist = dist
        print(f"The dist has been loaded from {self.args.dist_path}")
        return feature1, feature2
    
    def test(self, *args, **kwargs):
        return self.test_sp(*args, **kwargs)
    
    def load_two_features_and_dist(self, args, norm=False):
        if args.k != -1:
            args.k1 = args.k
            args.k2 = args.k
        args.feature1_path = f"shortestpath_cross/shortest_path_{args.gnn_model1}Encoder_{args.loss_type1}_me.pt"
        args.feature2_path = f"shortestpath_cross/shortest_path_{args.gnn_model2}Encoder_{args.loss_type2}_me.pt"
        # load feature
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

        if args.feature3_path.endswith(".npy"):
            dist = np.load(os.path.join(args.saved_features_dir, args.dist_path))
        else:
            dist = torch.load(os.path.join(args.saved_features_dir, args.dist_path), map_location="cpu")
            dist = dist.numpy()
        
        feature1 = feature1.reshape(feature1.shape[0], -1)
        feature2 = feature2.reshape(feature2.shape[0], -1)
        dist = dist.reshape(dist.shape[0], -1)
        if norm:
            feature1 = feature1 / np.linalg.norm(feature1, axis=1, keepdims=True)
            feature2 = feature2 / np.linalg.norm(feature2, axis=1, keepdims=True)
            # feature3 = feature3 / np.linalg.norm(feature3, axis=1, keepdims=True)
        print(f"【feature 1】: {compute_ae_loss(torch.from_numpy(feature1), torch.from_numpy(dist))}")
        print(f"【feature 2】: {compute_ae_loss(torch.from_numpy(feature2), torch.from_numpy(dist))}")
        return feature1, feature2, dist

    def test_sp(self, args, feature1, feature2, feature1_transformed, only_1_index, only_2_index, overlap_index, dist=None, normalization="max"):
        if dist is None:
            dist = self.dist
        feature1 = torch.from_numpy(feature1)
        feature2 = torch.from_numpy(feature2)
        feature1_transformed = torch.from_numpy(feature1_transformed)
        dist = torch.from_numpy(dist.copy())

        dist1 = torch.cdist(feature1, feature1, p=2)
        dist2 = torch.cdist(feature2, feature2, p=2)
        dist1_selected = torch.cdist(feature1_transformed, feature1_transformed, p=2)
        dist12 = torch.cdist(feature1, feature2, p=2)
        dist3 = torch.cdist(feature1_transformed, feature2, p=2)

        if normalization == "max" or normalization == "fix":
            dist1 = (dist1 / dist1.max() * dist.max())
            dist2 = (dist2 / dist2.max() * dist.max())
            dist1_selected = (dist1_selected / dist1_selected.max() * dist.max())
            dist12 = (dist12 / dist12.max() * dist.max())
            dist3 = (dist3 / dist3.max() * dist.max())
        elif normalization == "unnorma":
            max1_value = 1
            max2_value = 1
            max2_selected_value = 1
            max3 = 1
        
        return {"ae_dist1": cal_ae_loss(dist1, dist), "ae_dist2": cal_ae_loss(dist2, dist), "ae_dist2_selected": cal_ae_loss(dist1_selected, dist), 
                "ae_dist21": cal_ae_loss(dist12, dist),
                "ae_dist3": cal_ae_loss(dist3, dist),
                "mse_dist1": compute_mse_loss(dist1, dist), "mse_dist2": compute_mse_loss(dist2, dist), "mse_dist2_selected": compute_mse_loss(dist1_selected, dist),
                "mse_dist21": compute_mse_loss(dist12, dist), "mse_dist3": compute_mse_loss(dist3, dist),
                "me_dist1": cal_me_loss(dist1, dist), "me_dist2": cal_me_loss(dist2, dist), "me_dist2_selected": cal_me_loss(dist1_selected, dist),
                "me_dist21": cal_me_loss(dist12, dist), "me_dist3": cal_me_loss(dist3, dist),}