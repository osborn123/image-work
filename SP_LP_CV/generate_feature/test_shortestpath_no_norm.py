import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, VGAE
from torch_geometric.utils import train_test_split_edges, to_networkx
import networkx as nx
from tqdm import trange

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

def compute_shortest_path_distances(data):
    # Convert to NetworkX graph to compute shortest paths
    G = to_networkx(data, to_undirected=True)
    sp_dist = nx.floyd_warshall_numpy(G)
    sp_dist[sp_dist == float('inf')] = 0  # Replace infinity with 0 for normalization
    sp_dist = torch.tensor(sp_dist, dtype=torch.float)
    return sp_dist/sp_dist.max(), sp_dist

def compute_ae_loss(z, data, normalize="max"):
    sp_dist = data.sp_dist_raw
    if normalize == "max":
        dist = torch.cdist(z, z, p=2)
        dist = dist / dist.max()
        dist *= sp_dist.max()
    ae_loss = torch.abs(sp_dist - dist).max()
    return ae_loss

def compute_shortest_path_loss(z, data):
    # Compute pairwise distance in the embedding space
    dist = torch.cdist(z, z, p=2)
    # data.edge_index = data.train_pos_edge_index
    sp_dist = data.sp_dist_raw

    # Normalize distances
    # dist = dist / dist.max()

    # Compute the loss (MSE between normalized distances)
    loss = F.mse_loss(dist, sp_dist)
    return loss

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = compute_shortest_path_loss(z, data)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    return z, compute_ae_loss(z, data)

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

    for Encoder in [GCNEncoder, GATEncoder]:
        # Model
        model = VGAE(Encoder(data.num_features, 128)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        min_loss = np.inf
        for epoch in trange(1, 10000):
            loss = train(model, data, optimizer)
            z, ae_loss = test(model, data)
            print(f"AE Loss: {ae_loss} MIN Loss: {min_loss}")
            if ae_loss < min_loss:
                min_loss = ae_loss
                torch.save(z, f'./saved_features/shortestpath/shortest_path_{Encoder.__name__}_no_norm.pt')
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    torch.save(data.sp_dist_raw, f'./saved_features/shortestpath/native_dis_no_norm.pt')
    # torch.save(data.sp_dist_raw, f'./saved_features/shortestpath/native_dis_raw.pt')
