import torch, os
import copy
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, VGAE, SAGEConv
from torch_geometric.utils import train_test_split_edges, to_networkx
import networkx as nx
from tqdm import trange

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGEEncoder, self).__init__()
        # 使用两倍的输出通道数增加模型的容量
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        # 两个输出层，一个用于均值，一个用于logstd
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一层卷积后使用ReLU激活函数
        x = F.relu(self.conv1(x, edge_index))
        # 输出均值和log标准差
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv_mu = GATConv(2 * out_channels, out_channels)
        self.conv_logstd = GATConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def compute_shortest_path_distances(data, cache=True):
    # Convert to NetworkX graph to compute shortest paths
    G = to_networkx(data, to_undirected=True)
    if cache and os.path.exists(f'./saved_features/shortestpath/native_dis.pt'):
        sp_dist = torch.load(f'./saved_features/shortestpath/native_dis.pt')
    else:
        if not os.path.exists(f'./saved_features/shortestpath/native_dis.pt'):
            os.makedirs(f'./saved_features/shortestpath', exist_ok=True)
        sp_dist = nx.floyd_warshall_numpy(G)
        sp_dist[sp_dist == float('inf')] = 0  # Replace infinity with 0 for normalization
        sp_dist = torch.tensor(sp_dist, dtype=torch.float)
        torch.save(sp_dist, f'./saved_features/shortestpath/native_dis.pt')
    return sp_dist/sp_dist.max(), sp_dist

def compute_shortest_path_loss(z, data, loss_type="MSE"):
    # Compute pairwise distance in the embedding space
    dist = torch.cdist(z, z, p=2)
    # data.edge_index = data.train_pos_edge_index
    sp_dist = data.sp_dist

    # Normalize distances
    dist = dist / dist.max()

    # Compute the loss (MSE between normalized distances)
    if loss_type == "MSE":
        loss = F.mse_loss(dist, sp_dist)
    elif loss_type == "MAE":
        loss = F.l1_loss(dist, sp_dist)
    elif loss_type == "MAX":
        loss = torch.abs(sp_dist - dist).max()
    elif loss_type == "MIN":
        loss = torch.abs(sp_dist - dist).min()
    elif loss_type == "SUM":
        loss = torch.abs(sp_dist - dist).sum()
    elif loss_type == "MEAN":
        loss = torch.abs(sp_dist - dist).mean()
    return loss

def compute_me_loss(z, sp_dist, normalize="max"):
    # sp_dist = data.sp_dist_raw
    if normalize == "max":
        dist = torch.cdist(z, z, p=2)
        dist = dist / dist.max()
        dist *= sp_dist.max()
    me_loss = torch.abs(sp_dist - dist).mean()
    return me_loss

def train(model, data, optimizer, loss_type="MSE"):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = compute_shortest_path_loss(z, data, loss_type=loss_type)
    # print(f"me Loss: {compute_me_loss(z, data)}")
    loss.backward()
    optimizer.step()
    return z.detach(), loss.item()

def test(model, data, loss_type="MSE"):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    return z, compute_me_loss(z, data.sp_dist_raw), compute_shortest_path_loss(z, data, loss_type=loss_type).item()

def load_data(data="cora"):
    dataset = Planetoid(root='./data', name=data)
    data = dataset[0]
    data.sp_dist, data.sp_dist_raw = compute_shortest_path_distances(data)
    data = train_test_split_edges(data)
    return data

if __name__ == '__main__':
    data = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    for Encoder in [GCNEncoder, GATEncoder, GraphSAGEEncoder]:
        # Model
        for Loss_type in ["MSE", "MAE", "MAX", "MIN", "SUM", "MEAN"]:
            model = VGAE(Encoder(data.num_features, 128)).to(device)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
            min_loss = np.inf
            # improvement_threshold = 0.01  # 设置为1%的改进门槛
            best_model = None
            for epoch in trange(1, 10000):
                # loss = train(model, data, optimizer, loss_type="MAX")
                z, loss = train(model, data, optimizer, loss_type=Loss_type)
                _, me_loss, test_loss = test(model, data)
                # if me_loss < min_loss:
                #     min_loss = me_loss
                #     best_model = copy.deepcopy(z)
                if loss < min_loss:
                    min_loss = loss
                    best_model = copy.deepcopy(z)
                if epoch % 10 == 0:
                    print(f"me Loss: {me_loss} MIN Loss: {min_loss}")
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            # torch.save(best_model, f'./saved_features/shortestpath/shortest_path_{Encoder.__name__}_{Loss_type}_me_{min_loss:.4f}.pt')
            torch.save(best_model, f'./saved_features/shortestpath_cross/shortest_path_{Encoder.__name__}_{Loss_type}_me.pt')
            print("Saving to file: ", f'./saved_features/shortestpath_cross/shortest_path_{Encoder.__name__}_{Loss_type}_me.pt')
            print(f"Test:{compute_me_loss(best_model, data.sp_dist_raw)}")
    torch.save(data.sp_dist_raw, f'./saved_features/shortestpath_cross/native_dis.pt')
    # torch.save(data.sp_dist_raw, f'./saved_features/shortestpath/native_dis_raw.pt')
