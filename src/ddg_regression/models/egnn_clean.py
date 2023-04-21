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
from base.dataset import ddgData, ddgDataSet

from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_mean

import pandas as pd


def to_np(x): # returns tensor detached, on cpu and as numpy array
    return x.cpu().detach().numpy()

def zeros(tensor): # fills tensor with 0s
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor): # fills tensor with 1s
    if tensor is not None:
        tensor.data.fill_(1)

class GraphNorm(torch.nn.Module):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super(GraphNorm, self).__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """"""
        
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)[batch]
        out = x - mean * self.mean_scale
        var = scatter_mean(out.pow(2), batch, dim=0, dim_size=batch_size)
        std = (var + self.eps).sqrt()[batch]
        return self.weight * out / std + self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class ddgEGNN(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        loader_config: dict,
        dataset_config: dict,
        trainer_config: dict,
        num_edge_features: int = 0,
        embedding_in_nf: int = 32, # num features of embedding in
        embedding_out_nf: int = 32, # num features of embedding out
        egnn_layer_hidden_nfs: list = [32, 32, 32], # number and dimension of hidden layers; default 3x 32-dim layers
        num_classes: int=1, # dimension of output
        opt: str='adam', # optimizer
        loss: str='mse', # loss function
        scheduler: str=None,
        lr: float=10e-3, # learning rate
        dropout: float=0.0,
        balanced_loss: bool = False,
        attention: bool = False,
        residual: bool = True,
        normalize: bool = False,
        tanh: bool = False,
        update_coords: bool = True,
        weight_decay: float = 0,
        norm: str = None,
        norm_nodes: str = None,
        pool_graphvectors: str='diff', #None,max,mean
        **kwargs,
    ):
        super(ddgEGNN, self).__init__()

        # load parameters from yaml config file
        self.loader_config = loader_config
        self.dataset_config = dataset_config
        self.trainer_config = trainer_config
        self.update_coords = update_coords

        # this is for the inheriting models
        self.embedding_out_nf = embedding_out_nf
        self.num_classes = num_classes

        self.embedding_in = Linear(num_node_features, embedding_in_nf)
        self.embedding_out = Linear(embedding_in_nf, embedding_out_nf)

        self.dropout = dropout
        
        self.pool_graphvectors = pool_graphvectors
        print("pool_graphvectors parameter set to: ", pool_graphvectors)

        if not pool_graphvectors: # ie no pooling
            self.post_pool_linear = Linear(2*self.embedding_out_nf, self.embedding_out_nf)
        self.post_pool_linear_2 = Linear(self.embedding_out_nf, self.num_classes)

        
        egnn_layers = []
        for hidden_nf in egnn_layer_hidden_nfs:
            layer = E_GCL(
                embedding_in_nf, 
                hidden_nf, 
                embedding_in_nf, 
                edges_in_d=num_edge_features,
                act_fn=nn.SiLU(),
                attention=attention,
                residual=residual,  # if True, in and out nf need to be equal
                normalize=normalize,
                coords_agg='mean',
                tanh=tanh,
                norm_nodes=norm_nodes,
            )
            egnn_layers.append(layer)
        self.egnn_layers = ModuleList(egnn_layers)

        # setup of training environment
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        if loss == 'mse':
            self.loss_fn = MSELoss()
        else:
            raise NotImplementedError
        if balanced_loss:
            raise NotImplementedError

        self.test_set_predictions = []

        self.norm_nodes = norm_nodes
        if norm_nodes:
            self.graphnorm = GraphNorm(embedding_out_nf)

            
    def forward(self, graph):
        """
        Pass graph through EGNN layers and embeddings.
        Final output is difference between wt and mut graph embeddings.
        
        Args:
            graph from DataLoader (batch of graphs???)
        """
        graph_vectors = []
        for g_ind in range(2):
            # extract information from wt (g_ind=0) and mut (g_ind=1) graphs
            nodes = graph[f'x_{g_ind}'].float() # node features: whether node is on pr1/pr2, one-hot encoded type
            edge_ind = graph[f'edge_index_{g_ind}'] # indices of source & destination nodes defining edges
            coords = graph[f'pos_{g_ind}'].float() # node xyz coordinates
            edge_attr = graph[f'edge_attr_{g_ind}'].float() # whether edge is intra (0) or inter (1) protein
            
            nodes = self.embedding_in(nodes) # output tensor of dimension n_nodes x 128

            for egnn_layer in self.egnn_layers:
                edge_ind_post_dropout, edge_attr_post_dropout = dropout_adj(edge_ind, edge_attr=edge_attr, p=self.dropout) # randomly drops edges from adjacency matrix; default dropout probability (p) set to 0

                # update node features (and, if update_coords=True, coordinates)
                if self.update_coords:
                    nodes, coords, _ = egnn_layer(nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout, batch=graph[f'x_{g_ind}_batch'])
                else:
                    nodes, _, _ = egnn_layer(nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout, batch=graph[f'x_{g_ind}_batch'])

            if self.norm_nodes:
                nodes = self.graphnorm(relu(self.embedding_out(nodes)), graph[f'x_{g_ind}_batch'])

            graph_vectors.append(global_max_pool(nodes, graph[f'x_{g_ind}_batch'])) # takes maximum value from each column in nodes matrix to output 1x128 matrix (from n_nodes x 128 matrix)
        
        if not self.pool_graphvectors: # concatenate but don't pool graph_vectors -> relu activation function
            out = torch.cat(graph_vectors, dim=1)
            out = relu(self.post_pool_linear(out))
            print(out)
        else:
            graph_vectors = torch.stack(graph_vectors, dim=0)
            if self.pool_graphvectors == 'max':
                out = torch.amax(graph_vectors, dim=0)
            elif self.pool_graphvectors == 'mean':
                out = torch.mean(graph_vectors, dim=0)
            if self.pool_graphvectors == 'diff': # take difference between graph_vectors (1x128 matrix from global_max_pool) of wt and mut
                out = graph_vectors[0] - graph_vectors[1] # wt - mut
            #else:
            #    print('error, graphvector pooling function not implemented')

        print(out)
        out = self.post_pool_linear_2(out)

        # out = self.post_pool_linear(out)
        return out
    

    def training_step(self, batch, batch_idx):
        """
        Run batch through forward and return loss, prediction and true label
        
        Args:
            batch: N sets of aggregated graphs (for batch size N)
            batch_idx: batch index
        """
        
        y = batch.y # true ddG value

        pred = self.forward(batch) # run batch through forward
        if y.shape != pred.shape: # reshape labels if needed
            try:
                y = y.view_as(pred)
            except:
                print('Error in shape of labels vs pred')
        loss = self.loss_fn(pred, y.float()) # calculate loss
        self.log('train_loss', loss) # log loss

        return {'loss': loss, 'pred': pred, 'y': y}

    
    def validation_step(self, batch, batch_idx):
        """
        Run batch through forward and return loss, prediction and true label - validation
        
        Args:
            batch: N sets of aggregated graphs (for batch size N)
            batch_idx: batch index
        """

        y = batch.y # true ddG value

        pred = self.forward(batch) # run batch through forward
        if y.shape != pred.shape: # reshape labels if needed
            y = y.view_as(pred)
        loss = self.loss_fn(pred, y.float()) # calculate loss
        self.log('val_loss', loss) # log loss

        return {'loss': loss, 'pred': pred, 'y': y}

    
    def test_step(self, batch, batch_idx):
        """
        Run batch through forward and return loss, prediction and true label; save this information - test
        
        Args:
            batch: N sets of aggregated graphs (for batch size N)
            batch_idx: batch index
        """
        
        y = batch.y # true ddG value

        pred = self.forward(batch) # run batch through forward

        if y.shape != pred.shape: # reshape labels if needed
            y = y.view_as(pred)
        loss = self.loss_fn(pred, y.float()) # calculate loss
        #pred = sigmoid(pred)

        # save predicted output (along with wt and mut pdb, predicted score and true label)
        output_preds = []
        labels = to_np(y).flatten()
        for ind, score in enumerate(to_np(pred.flatten())):
            output_preds.append((batch.pdb_wt[ind], batch.pdb_mut[ind], score, labels[ind]))
        self.test_set_predictions += output_preds
        self.log('test_loss', loss)

        return {'loss': loss, 'pred': pred, 'y': y}
    

    def epoch_metrics(self, epoch_output):
        """
        Evaluate model performance
        
        Args:
            epoch_output
        """
        preds = []
        ys = []
        for step in epoch_output:
            pred = to_np(step['pred'].flatten()) # predicted value
            y = to_np(step['y'].flatten()) # true value
            preds += [i for i in pred]
            ys += [i for i in y]

        # calculate pearson correlation
        pearson_corr = pearsonr(ys, preds) # y_true=ys, y_score=preds

        return pearson_corr[0]

    
    def test_epoch_end(self, output):
        """
        At end of test epoch, calculate and log pearson corr
        
        Args:
            output
        """
        pearson_corr = self.epoch_metrics(output)
        print('test_pearson_corr', pearson_corr)
        self.log('test_pearson_corr', pearson_corr)
        
        
    def validation_epoch_end(self, output):
        """
        At end of val epoch, calculate and log pearson corr
        
        Args:
            output
        """
        pearson_corr = self.epoch_metrics(output)
        print('val_pearson_corr', pearson_corr)
        self.log('val_pearson_corr', pearson_corr)

        
    def training_epoch_end(self, output):
        """
        At end of train epoch, calculate and log pearson corr
        
        Args:
            output
        """
        pearson_corr = self.epoch_metrics(output)
        print('train_pearson_corr', pearson_corr)
        self.log('train_pearson_corr', pearson_corr)
        
        
    def save_test_predictions(self, filename: Path):
        """
        Save test predictions to csv file
        
        Args:
            filename (Path): path to output file
        """
        with open(filename, 'w') as outf:
            outf.write('wt_pdb,mut_pdb,pred_score,true_label\n')
            for p in self.test_set_predictions:
                outf.write(f'{p[0]},{p[1]},{p[2]},{p[3]}\n')

                
    def configure_optimizers(self):
        """ Configure optimizers and define scheduler """
        if self.opt == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        if not self.scheduler:
            return optimizer
        else:
            # define scheduler
            if self.scheduler == 'CosineAnnealingWarmRestarts':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    self.trainer_config['max_epochs'],
                    eta_min=1e-4)
            elif self.scheduler == 'CosineAnnealing':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    self.trainer_config['max_epochs'])
            else: 
                raise NotImplementedError
            return {
                'optimizer': optimizer,
                'scheduler': scheduler,
            }

        
    def test_dataloader(self):
        """ Load and batch test data """
        if self.dataset_config['input_files']['test'] is None:
            return None

        # initialize data loading with values from config file
        ds = ddgDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
        )
        # populate dataset with input csv file
        for f in self.dataset_config['input_files']['test']:
            ds.populate(f, overwrite=False)

        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'], 
            shuffle=False, # if True, data shuffled at every epoch
            num_workers=self.loader_config['num_workers'],
            follow_batch=batch_ls) # follow_batch enables mini-batching ie aggregating wt & mut graphs
        return loader

    
    def train_dataloader(self):
        """ Load and batch train data """
        if self.dataset_config['input_files']['train'] is None:
            return None
        
        # initialize data loading with values from config file
        ds = ddgDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames']
        )
        # populate dataset with input csv file
        for f in self.dataset_config['input_files']['train']:
            ds.populate(f, overwrite=False)

        if self.loader_config['balanced_sampling']:
            raise NotImplementedError('Balanced sampling not implemented.')
#             weights = ds.get_sample_weights()
#             sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        else:
            sampler = None

        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'],  
            shuffle=False, # if True, data shuffled at every epoch
            num_workers=self.loader_config['num_workers'],
            sampler=sampler,
            follow_batch=batch_ls) # follow_batch enables mini-batching ie aggregating wt & mut graphs
        
        return loader

    
    def val_dataloader(self):
        """ Load and batch val data """
        if self.dataset_config['input_files']['val'] is None:
            return None
        
        # initialize data loading with values from config file
        ds = ddgDataSet(
            interaction_dist=self.dataset_config['interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
        )
        # populate dataset with input csv file
        for f in self.dataset_config['input_files']['val']:
            ds.populate(f, overwrite=False)

        batch_ls = ['x_0', 'x_1']
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'], 
            shuffle=False, # if True, data shuffled at every epoch
            num_workers=self.loader_config['num_workers'],
            follow_batch=batch_ls) # follow_batch enables mini-batching ie aggregating wt & mut graphs
        
        return loader 


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


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
    print(h, x)
    
    

