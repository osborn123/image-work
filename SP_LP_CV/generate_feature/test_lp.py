"""
=> Test the link prediction with graph neural network
"""
import torch, os
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, VGAE
from torch_geometric.utils import train_test_split_edges, negative_sampling
from typing import Tuple

# 加载Cora数据集
def load_data(data="cora", device="cpu"):
    dataset = Planetoid(root='../data/', name=data)
    data = dataset[0]
    data = train_test_split_edges(data)
    return data.to(device)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def decode(z: Tensor, edge_index: Tensor, z_t: Tensor = None, sigmoid: bool = True) -> Tensor:
    if z_t is None:
        z_t = z
    value = (z[edge_index[0]] * z_t[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(value) if sigmoid else value

def test_lp(z: Tensor, pos_edge_index: Tensor,
            neg_edge_index: Tensor, z_target: Tensor = None) -> Tuple[Tensor, Tensor]:
    r"""Given latent variables :obj:`z`, positive edges
    :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
    computes area under the ROC curve (AUC) and average precision (AP)
    scores.

    Args:
        z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (torch.Tensor): The positive edges to evaluate
            against.
        neg_edge_index (torch.Tensor): The negative edges to evaluate
            against.
    """
    if z_target is None:
        z_target = z
    from sklearn.metrics import average_precision_score, roc_auc_score

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = decode(z, pos_edge_index, z_target, sigmoid=True)
    neg_pred = decode(z, neg_edge_index, z_target, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)

def recon_loss(model, z, pos_edge_index, neg_edge_index=None):
    EPS = 1e-15
    pos_loss = -torch.log(
        decode(z, pos_edge_index, sigmoid=True) + EPS).mean()

    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 - decode(z, neg_edge_index, sigmoid=True) + EPS).mean()

    loss = pos_loss + neg_loss
    return loss

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv_mu = GATConv(2 * out_channels, out_channels)
        self.conv_logstd = GATConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    # loss = model.recon_loss(z, data.train_pos_edge_index)
    loss = recon_loss(model, z, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    return test_lp(z, data.test_pos_edge_index, data.test_neg_edge_index)

def train_model(encoder_name="GAT", verbose=False):
    device = f"cuda:0"
    data = load_data(device=device)
    data_name = "cora"
    # save data
    torch.save(data, f'../saved_features/{data_name}.pt')
    # for Encoder, encoder_name in zip([GCNEncoder, GATEncoder], ["GCN", "GAT"]):
    Encoder = GCNEncoder if encoder_name == "GCN" else GATEncoder
    # VGAE Model
    max_aux = 0.0
    model = VGAE(Encoder(data.num_features, 128)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        loss = train(model, data, optimizer)
        auc, ap = test(model, data)
        if verbose:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
        if auc > max_aux and epoch > 20:
            max_aux = auc
            # save the best model
            if not os.path.exists(f'../saved_features/link_prediction/'):
                os.makedirs(f'../saved_features/link_prediction/')
            torch.save(model.encode(data.x, data.train_pos_edge_index).detach().cpu(), f'../saved_features/link_prediction/{encoder_name}_{data_name}.pt')
    return data, model

# 主函数
if __name__ == '__main__':
    train_model("GAT", verbose=True)
    train_model("GCN")