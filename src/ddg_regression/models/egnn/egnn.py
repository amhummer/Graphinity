from torch import nn
import torch
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch.nn import Module, Linear, MSELoss, ModuleList #BCEWithLogitsLoss
from torch.nn.functional import relu
#from torch import sigmoid
from torch.utils.data import WeightedRandomSampler
import pytorch_lightning as pl
from torch import optim
from torch_geometric.utils import dropout_adj
from torch_geometric.data import DataLoader as GeoDataLoader
from pathlib import Path
import sys
# from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import pearsonr
from models.graphnorm.graphnorm import GraphNorm

from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_mean

import pandas as pd


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re

    Adapted from https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py (Satorras et al., 2019)
    """

    def __init__(
        self, 
        input_nf, 
        output_nf, 
        hidden_nf, 
        edges_in_d=0, 
        act_fn=nn.SiLU(), 
        residual=True, 
        attention=False, 
        normalize=False, 
        coords_agg='mean', 
        tanh=False,
        norm_nodes=None):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.norm_nodes = norm_nodes
        if norm_nodes:
            self.node_norm_layer1 = GraphNorm(hidden_nf)
        else:
            self.node_norm_layer1 = torch.nn.Identity()

        # if norm_edge_mlp:
        #     self.edge_norm_layer1 = GraphNorm(hidden_nf)
        #     self.edge_norm_layer2 = GraphNorm(hidden_nf)
        # else:
        #     self.edge_norm_layer1 = torch.nn.Identity()
        #     self.edge_norm_layer2 = torch.nn.Identity()

        # self.edge_mlp = nn.Sequential(
        #     nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
        #     act_fn,
        #     self.edge_norm_layer1,
        #     nn.Linear(hidden_nf, hidden_nf),
        #     act_fn)

        self.edge_mlp1 = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn
        )

        self.edge_mlp2 = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp1 = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn
        )

        self.node_mlp2 = nn.Sequential(
            nn.Linear(hidden_nf, output_nf))


        # self.node_mlp = nn.Sequential(
        #     nn.Linear(hidden_nf + input_nf, hidden_nf),
        #     act_fn,
        #     self.node_norm_layer1,
        #     nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


    def edge_model(self, source, target, radial, edge_attr, batch):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp1(out)
        # out = self.edge_norm_layer1(out, batch=batch)
        out = self.edge_mlp2(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        # out = self.edge_norm_layer2(out, batch=batch)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, batch):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp1(agg)
        if self.norm_nodes:
            out = self.node_norm_layer1(out, batch=batch)
        out = self.node_mlp2(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, batch)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.

        Adapted from https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py (Satorras et al., 2019)
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        # self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

