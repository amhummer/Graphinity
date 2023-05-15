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
from models.graphnorm.graphnorm import GraphNorm
from models.egnn.egnn import E_GCL, EGNN

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
    
    

