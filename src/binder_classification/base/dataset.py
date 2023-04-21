from pathlib import Path
import struct
import numpy as np
from scipy.spatial.distance import cdist
import torch as th
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import pandas as pd
import sys
from torch_geometric.utils import remove_self_loops
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from collections import defaultdict
import random

# import code needed for typing atoms
if __name__ == '__main__':
    import utils
    from atom_types import Typer
else:
    try:
        from . import utils
        from .atom_types import Typer
    except:
        from base import utils
        from base.atom_types import Typer



class dgDataSet(Dataset):

    def __init__(self,
        run_id: str = "",
        interaction_dist: float=4, # cutoff interaction distance for generating graphs, default 4 A
        graph_mode: str='int_cdrh3', # graph generation mode
        typing_mode='lmg', # mode for atom typing, default LitMolGrid (lmg); other option is res
        cache_frames: bool=False, 
        rough_search: bool=False,
        force_recalc: bool = False,
        **kwargs):

        self.run_id = run_id

        self.type_map = utils.get_type_map()
        if typing_mode == 'lmg':
            self.node_feature_size = 12  # 11 atom types and 1/0 for binding partner ID
        elif typing_mode == 'res_type':
            self.node_feature_size = 22  # 20 aa, 1 for non-canonical aa, 1/0 for binding partner ID
        self.ppi_defs = pd.DataFrame() # store information about entry
        self.labels = [] # dG labels
        self.edge_dim = 2  # intra- vs inter-protein edges
        self.interaction_dist = interaction_dist
        self.typing_mode = typing_mode # typing mode
        self.graph_mode = graph_mode # graph_generation mode 
        self.cache = {}
        self.cache_frames = cache_frames
        self.speedup = rough_search

        self.graph_generation_function_dict = { # options for graph generation
            'int_cdrh3': self._get_int_cdrh3_graph,
        }

        self.aa_map = { # map AAs to numbers
            'ALA': 0,
            'ARG': 1,
            'ASN': 2,
            'ASP': 3,
            'CYS': 4,
            'GLU': 5,
            'GLN': 6,
            'GLY': 7,
            'HIS': 8,
            'ILE': 9,
            'LEU': 10,
            'LYS': 11,
            'MET': 12,
            'PHE': 13,
            'PRO': 14,
            'SER': 15,
            'THR': 16,
            'TRP': 17,
            'TYR': 18,
            'VAL': 19,
            'XXX': 20,
        }

        
    def _get_int_cdrh3_graph(self, pr1_df: pd.DataFrame, pr2_df: pd.DataFrame):
        """Return graph composed of nodes in cdrh3 and surrounding neighborhood nodes (inter- and intra-protein edges)

        Args:
            pr1_df (pd.DataFrame): Protein 1 dataframe (ab)
            pr2_df (pd.DataFrame): Protein 2 dataframe (ag)
        """

        # obtain coordinates of pr1, pr2 and the mutation site (on pr1)
        coords_pr1 = pr1_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_pr2 = pr2_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)
        coords_cdrh3 = pr1_df.loc[(pr1_df["chain_id"] == "B") & (pr1_df["residue_number"].isin(np.arange(99,109))), ['x_coord', 'y_coord', 'z_coord']].to_numpy(dtype=np.float64)

        ### ID NODES ####
        # dist between mutation and points on same protein (antibody, pr1)
        dist_between_cdrh3_pr1 = cdist(coords_cdrh3, coords_pr1)

        # identifying nodes within distance of mut on same protein
        _, intra_pr1 = np.where(dist_between_cdrh3_pr1 < self.interaction_dist) # _ is num of mut_site_df
        pr1_node_idx = sorted(np.unique(intra_pr1))
        pr1_nodes = pr1_df.iloc[pr1_node_idx, :]
        coords_pr1 = coords_pr1[pr1_node_idx, :] # overwrite coords_pr1 with nodes only
        
        # dist between pr1 nodes (mut site + everything within distance) and points on protein 2 (antigen)
        dist_between_pr1_nodes_pr2 = cdist(coords_pr1, coords_pr2)

        # identifying nodes within distance of pr1 nodes
        _, inter_pr2 = np.where(dist_between_pr1_nodes_pr2 < self.interaction_dist)
        pr2_node_idx_init = sorted(np.unique(inter_pr2))
        pr2_nodes_init = pr2_df.iloc[pr2_node_idx_init, :]
        coords_pr2_init = coords_pr2[pr2_node_idx_init, :]

        # dist between pr1 nodes (mut site + everything within distance) and points on protein 2 (antigen)
        dist_between_pr2_nodes_pr2 = cdist(coords_pr2_init, coords_pr2)

        # identifying nodes within distance of pr2 nodes
        _, intra_pr2 = np.where(dist_between_pr2_nodes_pr2 < self.interaction_dist)
        pr2_node_idx = sorted(np.unique(intra_pr2))
        pr2_nodes = pr2_df.iloc[pr2_node_idx, :]
        coords_pr2 = coords_pr2[pr2_node_idx, :] # overwrite coords_pr2 with nodes only

        #### MAKE FEATURE VECTOR ####
        # features: one-hot encoded lmg atome type
        node_features_pr1 = self._get_node_features(pr1_nodes)
        node_features_pr2 = self._get_node_features(pr2_nodes)

        # is the node on pr1 or pr2?
        node_bool_pr1 = np.zeros((len(coords_pr1), 1))
        node_bool_pr2 = np.ones((len(coords_pr2), 1))

        # features: node xyz coordinates [cols 0-2], one-hot encoded lmg types [cols 3-13], label of whether pr1/pr2 [col 14]
        feats_pr1 = np.concatenate([coords_pr1, node_features_pr1, node_bool_pr1], axis=-1)
        feats_pr2 = np.concatenate([coords_pr2, node_features_pr2, node_bool_pr2], axis=-1)

        feats = np.concatenate([feats_pr1, feats_pr2], axis=-2)

        #### MAKE EDGE LIST ####
        
        # 1 inter: interactions between pr1 & pr2
        dist_inter = cdist(coords_pr1, coords_pr2)
        inter_src_pr1, inter_dst_pr2 = np.where(dist_inter < self.interaction_dist)

        # 2 intra_pr1: interactions between pr1 nodes
        dist_intra_pr1 = cdist(coords_pr1, coords_pr1)
        intra_src_pr1, intra_dst_pr1 = np.where(dist_intra_pr1 < self.interaction_dist)

        # 3 intra_pr2: interactions between pr2 nodes
        dist_intra_pr2 = cdist(coords_pr2, coords_pr2)
        intra_src_pr2, intra_dst_pr2 = np.where(dist_intra_pr2 < self.interaction_dist)

        # don't need node lookup as only final distances (for edges) only measured for nodes in final dataset 
        pr2_offset = len(pr1_nodes)

        # edge source and destination nodes
        edge_src = np.concatenate([
            [i for i in inter_src_pr1],
            [i for i in intra_src_pr1],
            [i + pr2_offset for i in intra_src_pr2]
        ])
        edge_dst = np.concatenate([
            [i + pr2_offset for i in inter_dst_pr2],
            [i for i in intra_dst_pr1],
            [i + pr2_offset for i in intra_dst_pr2]
        ])

        # each edge must be bidirectional
        edge_src_full = np.concatenate([edge_src, edge_dst])
        edge_dst_full = np.concatenate([edge_dst, edge_src])
        edge_indices = np.vstack([edge_src_full, edge_dst_full]).astype(np.float64)

        #### MAKE EDGE ATTRIBUTES ####

        # for now just 1/0 for inter/intra
        edge_attr = np.concatenate([
            np.ones(len(inter_dst_pr2)),
            np.zeros(len(intra_dst_pr1)),
            np.zeros(len(intra_dst_pr2)),
        ])
        
        edge_attr_full = np.expand_dims(np.concatenate([edge_attr, edge_attr]), 1)

        return feats, edge_indices, edge_attr_full

    def _get_node_features(self, df: pd.DataFrame):
        mode = self.typing_mode

        if mode == 'lmg':
            types = df['lmg_types'] # get lmg_types column (integers) from parquet df generated via typing in parse_pdb_to_parquet
            types = types.apply(lambda x: self.type_map[x]) # map to ? using type_map dictionary generated with utils.py get_type_map function
            types = np.array(types) # convert df column / series to np array
            types = utils.get_one_hot(types, nb_classes=max(self.type_map.values()) + 1) # one-hot encode
            return types
        elif mode == 'res_type':
            types = df['residue_name'] # get residue name
            types = types.apply(lambda x: self.aa_map[x] if x in self.aa_map.keys() else 20).astype(np.int64) # map to integer as defined in aa_map
            types = np.array(types)
            types = utils.get_one_hot(types, nb_classes=max(self.aa_map.values()) + 1) # one-hot encode
            return types
        else:
            raise NotImplementedError(mode)

            
    def _generate_graph_dict(self, ab_df: pd.DataFrame, ag_df: pd.DataFrame):
        """Generate dictionary of nodes, edge indices and edge attributes in graph

       Args:
            ab_df (pd.DataFrame): Protein 1 dataframe
            ag_df (pd.DataFrame): Protein 2 dataframe
        """
        graph_dict = {}
        nodes, edge_ind, edge_attr = self.graph_generation_function_dict[self.graph_mode](ab_df, ag_df)
        ## nodes = nodes with features: node xyz coordinates [cols 0-2], one-hot encoded types [cols 3-13], bool ab/2 [col 14] 
        ## edge_ind: indices of source & destination nodes defining edges
        ## edge_attr: label of whether intra- or inter-protein edge
        
        graph_dict['nodes'] = nodes
        graph_dict['edge_ind'] = edge_ind
        graph_dict['edge_attr'] = edge_attr

        return graph_dict

    
    def _parse_graph(self, graph_dict: dict, label:np.ndarray, ppi_def:dict):
        """Generate parsed graph object in correct format. This is intended to be overwritten 
            by child classes depending on requirements of the downstream models. 
            Base implementation parses the data into a pytorch geometric data instance.

        Args:
            graph_dict: output of _generate_graph_dict, contains nodes (coordinates & features), edge indices and edge attributes of graph
        """

        # remove self loops
        edge_index, edge_attr = remove_self_loops(
                edge_index=th.from_numpy(graph_dict['edge_ind']).long(),
                edge_attr=th.from_numpy(graph_dict['edge_attr'])
            )   
        
        graph = Data( # add information to torch_geometric.data.Data ("A data object describing a homogeneous graph")
            x=th.from_numpy(graph_dict['nodes'][:,3:]), # node features: whether node is on ab/ag, one-hot encoded type
            edge_index=edge_index, # source & destination nodes defining edges
            edge_attr=edge_attr, # whether edge is intra (0) or inter (1) protein
            pos=th.from_numpy(graph_dict['nodes'][:,:3]), # node xyz coordinates
            y=th.tensor(label), # ddG label
            pdb_file = str(ppi_def['pdb']) # path to pdb file
        )
        
        return graph
    
        
    def __len__(self):
        """ Get number of entries in dataset """
        return len(self.labels)


    def populate(self, input_file: Path, overwrite: bool=False): #changed  
        """Extract information from input files and save in ppi_defs list of lists

        Args:
            input_file: cvs containing columns: labels, pdb [path to file], ab_chains, ag_chains
        """
        
        ppi_defs = pd.read_csv(input_file)
        labels = ppi_defs["labels"].to_list()

        # either overwrite or add to self.ppi_defs and self.labels
        if overwrite:
            self.ppi_defs = ppi_defs
            self.labels = labels
        else:
            if len(self.ppi_defs) > 0:
                self.ppi_defs += ppi_defs
            else:
                self.ppi_defs = ppi_defs
            if len(self.labels) != 0:
                self.labels = self.labels + labels
            else:
                self.labels = labels
    

    def __getitem__(self, idx: int, force_recalc: bool = False):
        """ Generate graph for complex in dataset
        
        Args:
            idx: index
        """
        
        ##### read in pdbs #####

        # obtain label and ppi complex info (path to pdb file, chains in ab & ag)
        label = self.labels[idx]
        ppi_def = self.ppi_defs.iloc[idx]

        # generate path to typed parquet file
        typed_pdb = Path(str(ppi_def['pdb']).replace('.pdb', '.parquet'))
        
        # check if typed file in cache
        if self.cache_frames and str(typed_pdb) in self.cache and not force_recalc:
            pdb_df = self.cache[str(typed_pdb)].copy()

        # check if typed file exists
        elif typed_pdb.exists() and not force_recalc:
            pdb_df = pd.read_parquet(typed_pdb)
                            
        # if not create and save typed files
        else:
            pdb_df = utils.parse_pdb_to_parquet(ppi_def['pdb'], typed_pdb, lmg_typed=True, ca=False)


        if self.cache_frames:
            if not str(typed_pdb) in self.cache:
                self.cache[str(typed_pdb)] = pdb_df.copy()
        
        ##### generate graphs #####
        
        # split pdb file into ab/ag
        ## identify chains in ab/ag
        chs_ab = []
        for ch in ppi_def['ab_chains']:
            chs_ab.append(ch)

        chs_ag = []
        for ch in ppi_def['ag_chains']:
            chs_ag.append(ch)
        
        ab_df = pdb_df[pdb_df['chain_id'].isin(chs_ab)]
        ag_df = pdb_df[pdb_df['chain_id'].isin(chs_ag)]
            
        # generate graph
        graph_dict = self._generate_graph_dict(ab_df, ag_df)
        graph = self._parse_graph(graph_dict, label, ppi_def=ppi_def)

        return graph
