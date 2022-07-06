
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class GraphEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(GraphEncoder, self).__init__()
        self.gc_feat = GCNConv(input_dims, hidden_dims)
        self.gc_mean = GCNConv(hidden_dims, output_dims)
        self.gc_var = GCNConv(hidden_dims, output_dims)

    def forward(self, x, edge_index, edge_weight):
        x = self.gc_feat(x, edge_index, edge_weight).relu()
        mean = self.gc_mean(x, edge_index, edge_weight)
        var = self.gc_var(x, edge_index, edge_weight)
        return mean, var


def full_block(input_dims, output_dims, drop_rate=0.2):
    return nn.Sequential(
        nn.Linear(input_dims, output_dims),
        nn.BatchNorm1d(output_dims, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=drop_rate)
    )


class SpatialModel(nn.Module):
    def __init__(self, input_dims, gae_dims, dae_dims):
        super(SpatialModel, self).__init__()
        self.input_dims = input_dims
        self.gae_dims = gae_dims
        self.dae_dims = dae_dims
        self.feat_dims = self.dae_dims[1] + self.gae_dims[1]
        self.encoder = nn.Sequential(
            full_block(self.input_dims, self.dae_dims[0]),
            full_block(self.dae_dims[0], self.dae_dims[1])
        )
        self.decoder = full_block(self.feat_dims, self.input_dims)
        self.vgae = VGAE(GraphEncoder(self.dae_dims[1], self.gae_dims[0], self.gae_dims[1]))

    def forward(self, x, edge_index, edge_weight):
        feat_x = self.encoder(x)
        feat_g = self.vgae.encode(feat_x, edge_index, edge_weight)
        feat = torch.concat([feat_x, feat_g], 1)
        x_dec = self.decoder(feat)
        dae_loss = F.mse_loss(x_dec, x)
        gae_loss = self.recon_loss(feat, edge_weight, edge_index) + 1 / len(x) * self.vgae.kl_loss()
        return feat, dae_loss, gae_loss

    def recon_loss(self, z, edge_weight, pos_edge_index, neg_edge_index=None):
        pos_dec = self.vgae.decoder(z, pos_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(pos_dec, edge_weight)
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_dec = self.vgae.decoder(z, neg_edge_index)
        neg_loss = -F.logsigmoid(-neg_dec).mean()
        return pos_loss + neg_loss

