from torch import nn
import torch
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch.nn import Module, Linear, Softmax, BCEWithLogitsLoss, ModuleList
#from torch.nn import Module, Linear, MSELoss, ModuleList
from torch.nn.functional import relu
from torch import sigmoid
from torch.utils.data import WeightedRandomSampler
import pytorch_lightning as pl
from torch import optim
from torch_geometric.utils import dropout_adj
from torch_geometric.data import DataLoader as GeoDataLoader
from pathlib import Path
import sys
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc
#from scipy.stats import pearsonr
from base.dataset import dgDataSet
from models.graphnorm.graphnorm import GraphNorm
from models.egnn.egnn import E_GCL, EGNN
import numpy as np

from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_mean


def to_np(x):
    return x.cpu().detach().numpy()

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


class dgEGNN(pl.LightningModule):
    def __init__(
        self,
        num_node_features: int,
        loader_config: dict,
        dataset_config: dict,
        trainer_config: dict,
        run_id: str = "", # var1var2 input when running train.py
        num_edge_features: int = 0,
        embedding_in_nf: int = 32, # num features of embedding in
        embedding_out_nf: int = 32, # num features of embedding out
        egnn_layer_hidden_nfs: list = [32, 32, 32], # number and dimension of hidden layers; default 3x 32-dim layers
        num_classes: int=1, # dimension of output
        opt: str='adam', # optimizer
        loss: str='bce_logits', # loss function
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
        pooling: str = 'max',
        pool_first: bool = True, # pool before or after passing to last linear layer
        **kwargs,
    ):
        super(dgEGNN, self).__init__()

        self.run_id = run_id

        # load parameters from yaml config file
        self.loader_config = loader_config
        self.dataset_config = dataset_config
        self.trainer_config = trainer_config
        self.update_coords = update_coords

        try: self.masking_bool = self.dataset_config['masking']
        except: self.masking_bool = False

        # this is for the inheriting models
        self.embedding_out_nf = embedding_out_nf
        self.num_classes = num_classes

        # pooling
        self.pool_first = pool_first
        if pooling == 'max':
            self.pooling_fn = global_max_pool
        elif pooling == 'mean':
            self.pooling_fn = global_mean_pool
        else:
            raise NotImplementedError('Pooling function not implemented')

        self.embedding_in = Linear(num_node_features, embedding_in_nf)
        self.embedding_out = Linear(embedding_in_nf, embedding_out_nf)

        if pool_first:
            self.post_pool_linear = Linear(embedding_out_nf, num_classes)
        else:
            self.pre_pool_linear = Linear(embedding_out_nf, num_classes)
        self.dropout = dropout

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
        if loss == 'bce_logits':
            self.loss_fn = BCEWithLogitsLoss()
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
        
        Args:
            graph from DataLoader
        """
        
        # extract information from graph
        nodes = graph.x.float() # node features: whether node is on pr1/pr2, one-hot encoded type
        edge_ind = graph.edge_index # indices of source & destination nodes defining edges
        coords = graph.pos.float() # node xyz coordinates
        edge_attr = graph.edge_attr.float() # whether edge is intra (0) or inter (1) protein

        nodes = self.embedding_in(nodes)

        for egnn_layer in self.egnn_layers:
            edge_ind_post_dropout, edge_attr_post_dropout = dropout_adj(edge_ind, edge_attr=edge_attr, p=self.dropout) # randomly drops edges from adjacency matrix; default dropout probability (p) set to 0
            
            # update node features (and, if update_coords=True, coordinates)
            if self.update_coords:
                nodes, coords, _ = egnn_layer(nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout, batch=graph.batch)
            else:
                nodes, _, _ = egnn_layer(nodes, edge_ind_post_dropout, coords, edge_attr_post_dropout, batch=graph.batch)

        if self.norm_nodes:
            nodes = self.graphnorm(relu(self.embedding_out(nodes)), graph.batch)

        if self.pool_first:
            graph_vector = self.pooling_fn(nodes, graph.batch)
            out = self.post_pool_linear(graph_vector)
        else:
            nodes = self.pre_pool_linear(nodes)
            out = self.pooling_fn(nodes, graph.batch)

        return out

    def training_step(self, batch, batch_idx):
        """
        Run batch through forward and return loss, prediction and true label
        
        Args:
            batch: N sets of graphs (for batch size N)
            batch_idx: batch index
        """

        y = batch.y # true dg value

        batch_idx = sorted(batch.batch.unique())
        batch_repl = {}

        for i, b in enumerate(batch_idx):
            batch_repl[b.item()] = i
        for i in range(len(batch.batch)):
            b = batch.batch[i].item()
            batch.batch[i] = batch_repl[b]

        pred = self.forward(batch) # run batch through forward

        if y.shape != pred.shape: # reshape labels if needed
            y = y.view_as(pred)
        loss = self.loss_fn(pred.float(), y.float()) # calculate loss
        self.log('train_loss', loss)

        return {'loss': loss, 'pred': pred, 'y': y}

    def validation_step(self, batch, batch_idx):
        """
        Run batch through forward and return loss, prediction and true label - validation
        
        Args:
            batch: N sets of aggregated graphs (for batch size N)
            batch_idx: batch index
        """
        
        y = batch.y # true dg value
        
        batch_idx = sorted(batch.batch.unique())
        batch_repl = {}
        for i, b in enumerate(batch_idx):
            batch_repl[b.item()] = i
        for i in range(len(batch.batch)):
            b = batch.batch[i].item()
            batch.batch[i] = batch_repl[b]

        y = torch.masked_select(y, y != -1)
        pred = self.forward(batch) # run batch through forward

        if y.shape != pred.shape: # reshape labels if needed
            y = y.view_as(pred)
        loss = self.loss_fn(pred.float(), y.float()) # calculate loss
        self.log('val_loss', loss) # calculate loss

        return {'loss': loss, 'pred': pred, 'y': y}

    def test_step(self, batch, batch_idx):
        """
        Run batch through forward and return loss, prediction and true label; save this information - test
        
        Args:
            batch: N sets of aggregated graphs (for batch size N)
            batch_idx: batch index
        """
        y = batch.y # true dg value

        batch_idx = sorted(batch.batch.unique())
        batch_repl = {}
        for i, b in enumerate(batch_idx):
            batch_repl[b.item()] = i
        for i in range(len(batch.batch)):
            b = batch.batch[i].item()
            batch.batch[i] = batch_repl[b]

        pred = self.forward(batch) # run batch through forward

        if y.shape != pred.shape: # reshape labels if needed
            y = y.view_as(pred)
        loss = self.loss_fn(pred.float(), y.float()) # calculate loss
        pred = sigmoid(pred)

        # save predicted output (along with pdb, predicted score and true label)
        output_preds = []
        labels = to_np(y).flatten()
        for ind, score in enumerate(to_np(pred)):
            if self.masking_bool:
                output_preds.append((batch.pdb_file[ind], batch.masked_cdrh3_res[ind], score, labels[ind]))
            else:
                output_preds.append((batch.pdb_file[ind], score, labels[ind]))
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
            ys += [float(i) for i in y]

        try:
            roc = roc_auc_score(y_true=ys, y_score=preds)
            ap = average_precision_score(y_true=ys, y_score=preds)
            precision, recall, thresholds = precision_recall_curve(y_true=ys, probas_pred=preds)
            auc_pr = auc(recall, precision)
        except:
            roc = None
            ap = None
            auc_pr = None
        
        return roc, ap, auc_pr

    def test_epoch_end(self, output):
        """
        At end of test epoch, calculate and log roc
        
        Args:
            output
        """
        roc, ap, auc_pr = self.epoch_metrics(output)
        print('test_roc', roc)
        self.log('test_roc', roc)
        self.log('test_average_precision', ap)

        
    def validation_epoch_end(self, output):
        """
        At end of val epoch, calculate and log roc
        
        Args:
            output
        """
        roc, ap, auc_pr = self.epoch_metrics(output)
        print('val_roc', roc)
        self.log('val_roc', roc)
        self.log('val_average_precision', ap)
        
    def training_epoch_end(self, output):
        """
        At end of train epoch, calculate and log roc
        
        Args:
            output
        """
        roc, ap, auc_pr = self.epoch_metrics(output)
        print('train_roc', roc)
        self.log('train_roc', roc)
        self.log('train_average_precision', ap)
        
    def save_test_predictions(self, filename: Path):
        """
        Save test predictions to csv file
        
        Args:
            filename (Path): path to output file
        """
        with open(filename, 'w') as outf:
            if self.masking_bool:
                outf.write('pdb,masked_cdrh3_res,pred_score,true_label\n')
                for p in self.test_set_predictions:
                    outf.write(f'{p[0]},{p[1]},{p[2]},{p[3]}\n')
            else:
                outf.write('pdb,pred_score,true_label\n')
                for p in self.test_set_predictions:
                    outf.write(f'{p[0]},{p[1]},{p[2]}\n')

                
    def configure_optimizers(self):
        """ Configure optimizers and define scheduler """
        if self.opt == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
        ds = dgDataSet(
            run_id=self.run_id,
            inter_interaction_dist=self.dataset_config['inter_interaction_dist'],
            intra_interaction_dist=self.dataset_config['intra_interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
            masking=self.masking_bool
        )
        # populate dataset with input csv file
        for f in self.dataset_config['input_files']['test']:
            ds.populate(f, overwrite=False)
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'], 
            shuffle=False,
            num_workers=self.loader_config['num_workers'])
        return loader

    
    def train_dataloader(self):
        """ Load and batch train data """
        if self.dataset_config['input_files']['train'] is None:
            return None
        
        # initialize data loading with values from config file
        ds = dgDataSet(
            run_id=self.run_id,
            inter_interaction_dist=self.dataset_config['inter_interaction_dist'],
            intra_interaction_dist=self.dataset_config['intra_interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
            masking=self.masking_bool,
        )
        # populate dataset with input csv file
        for f in self.dataset_config['input_files']['train']:
            ds.populate(f, overwrite=False)

        if self.loader_config['balanced_sampling']:
            raise NotImplementedError('Balanced sampling not implemented.')
#             weights = ds.get_sample_weights()
#             sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
#             shuffle = False
        else:
            sampler = None
            shuffle = False
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'],  
            shuffle=shuffle, 
            num_workers=self.loader_config['num_workers'],
            sampler=sampler)
        
        return loader

    def val_dataloader(self):
        """ Load and batch val data """
        if self.dataset_config['input_files']['val'] is None:
            return None
                
        # initialize data loading with values from config file
        ds = dgDataSet(
            run_id=self.run_id,
            inter_interaction_dist=self.dataset_config['inter_interaction_dist'],
            intra_interaction_dist=self.dataset_config['intra_interaction_dist'],
            graph_mode=self.dataset_config['graph_generation_mode'],
            typing_mode=self.dataset_config['typing_mode'],
            cache_frames=self.dataset_config['cache_frames'],
            masking=self.masking_bool,
        )
        # populate dataset with input csv file
        for f in self.dataset_config['input_files']['val']:
            ds.populate(f, overwrite=False)
        loader = GeoDataLoader(
            ds, 
            batch_size=self.loader_config['batch_size'], 
            shuffle=False,
            num_workers=self.loader_config['num_workers'])
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
