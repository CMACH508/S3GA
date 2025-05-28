import copy
import os.path as osp
import sys
from typing import Optional

import torch
import torch.utils.data
from torch_sparse import SparseTensor, cat
from torch_geometric.utils import subgraph


class ClusterData(torch.utils.data.Dataset):
    r""" Clusters/partitions a graph data object into multiple subgraphs,
    according to the cluster_list.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        cluster_list (tuple): the cluster_idx for data.x1 and data.x2.
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
    """
    def __init__(self, data, num_parts: int, cluster_list: tuple, map_list: tuple=(None,None), 
                log: bool = True):
        self.num_parts = num_parts
        self.clus_src, self.clus_tgt = cluster_list
        self.src_map, self.tgt_map = map_list
        if log:
            print('Computing partitioning according cluster list...', file=sys.stderr)
        self.data = data

    def __getitem__(self, idx):
        data = copy.copy(self.data)
        data.x1 = data.x1[self.clus_src[idx]] if len(self.clus_src[idx]) > 0 else torch.tensor([])
        data.x2 = data.x2[self.clus_tgt[idx]] if len(self.clus_tgt[idx]) > 0 else torch.tensor([])

        data.edge_index1, data.rel1 = subgraph(subset=self.clus_src[idx],
                                               edge_index=data.edge_index1,
                                               edge_attr=data.rel1, relabel_nodes=True, num_nodes=max(self.clus_src[idx].max()+1, data.edge_index1.max()+1)) \
                                    if len(self.clus_src[idx]) > 0 else (torch.tensor([[]]), torch.tensor([]))
        data.edge_index2, data.rel2 = subgraph(subset=self.clus_tgt[idx],
                                               edge_index=data.edge_index2,
                                               edge_attr=data.rel2, 
                                               relabel_nodes=True, num_nodes=max(self.clus_tgt[idx].max()+1, data.edge_index2.max()+1)) \
                                    if len(self.clus_tgt[idx]) > 0 else (torch.tensor([[]]), torch.tensor([]))

        
        data.num_node1 = data.x1.size(0)
        data.num_node2 = data.x2.size(0)
        data.num_edge1 = data.edge_index1.size(1)
        data.num_edge2 = data.edge_index2.size(1)
        data.map1 = self.src_map[idx] if self.src_map is not None else torch.arange(self.clus_src[idx].size(0))
        data.map2 = self.tgt_map[idx] if self.tgt_map is not None else torch.arange(self.clus_tgt[idx].size(0))
        data.assoc1 = self.clus_src[idx]
        data.assoc2 = self.clus_tgt[idx]
        return data
    
    def __len__(self):
        return len(self.clus_src)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f')')
        
class EvalClusterData(torch.utils.data.Dataset):
    def __init__(self, data, num_parts: int, cluster_list: tuple, map_list: tuple=(None,None), 
                log: bool = True):
        self.num_parts = num_parts
        self.clus_src, self.clus_tgt = cluster_list
        self.src_map, self.tgt_map = map_list
        if log:
            print('Computing partitioning according cluster list...', file=sys.stderr)
        self.data = data
    
    
    def __getitem__(self, idx):
        data = copy.copy(self.data)
        data.x1 = data.x1[self.clus_src[idx][self.src_map[idx]]] if len(self.src_map[idx]) > 0 else torch.tensor([])
        data.x2 = data.x2[self.clus_tgt[idx][self.tgt_map[idx]]] if len(self.tgt_map[idx]) > 0 else torch.tensor([])

        data.edge_index1, data.rel1 = subgraph(subset=self.clus_src[idx][self.src_map[idx]],
                                               edge_index=data.edge_index1,
                                               edge_attr=data.rel1, relabel_nodes=True, num_nodes=max(self.clus_src[idx].max()+1, data.edge_index1.max()+1)) \
                                    if len(self.clus_src[idx]) > 0 else (torch.tensor([[]]), torch.tensor([]))
        data.edge_index2, data.rel2 = subgraph(subset=self.clus_tgt[idx][self.tgt_map[idx]],
                                               edge_index=data.edge_index2,
                                               edge_attr=data.rel2, 
                                               relabel_nodes=True, num_nodes=max(self.clus_tgt[idx].max()+1, data.edge_index2.max()+1)) \
                                    if len(self.clus_tgt[idx]) > 0 else (torch.tensor([[]]), torch.tensor([]))

        data.num_node1 = data.x1.size(0)
        data.num_node2 = data.x2.size(0)
        data.num_edge1 = data.edge_index1.size(1)
        data.num_edge2 = data.edge_index2.size(1)
        data.map1 = torch.arange(self.clus_src[idx].size(0))
        data.map2 = torch.arange(self.clus_tgt[idx].size(0))
        data.assoc1 = self.clus_src[idx][self.src_map[idx]] if len(self.src_map[idx]) > 0 else torch.tensor([]) 
        data.assoc2 = self.clus_tgt[idx][self.tgt_map[idx]] if len(self.tgt_map[idx]) > 0 else torch.tensor([])
        return data
    
    def __len__(self):
        return len(self.clus_src)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f')')

class SuperClusterData(torch.utils.data.Dataset):
    def __init__(self, data, num_parts: int, cluster_list: tuple, map_list: tuple=(None,None), 
                log: bool = True) -> None:
        super().__init__()
        self.num_parts = num_parts
        self.clus_src, self.clus_tgt = cluster_list
        self.src_map, self.tgt_map = map_list
        if log:
            print('Computing partitioning according cluster list...', file=sys.stderr)
        self.data = data

    def __compute_label__(self, clu_src, clu_tgt):
        row, col = self.data.gt_y
        src_mask = torch.isin(row, clu_src)
        tgt_mask = torch.isin(col, clu_tgt)
        gt_mask = src_mask & tgt_mask
        sub_row, sub_col = row[gt_mask], col[gt_mask]
        # relabel id
        src_idx = (sub_row.unsqueeze(1) == clu_src.unsqueeze(0)).nonzero(as_tuple=True)[1]
        tgt_idx = (sub_col.unsqueeze(1) == clu_tgt.unsqueeze(0)).nonzero(as_tuple=True)[1]
        
        return torch.stack([src_idx, tgt_idx], dim=0)
        

    def __getitem__(self, idx):
        data = copy.copy(self.data)
        data.x1 = data.x1[self.clus_src[idx]] if len(self.clus_src[idx]) > 0 else torch.tensor([])
        data.x2 = data.x2[self.clus_tgt[idx]] if len(self.clus_tgt[idx]) > 0 else torch.tensor([])

        data.edge_index1, data.rel1 = subgraph(subset=self.clus_src[idx],
                                               edge_index=data.edge_index1,
                                               edge_attr=data.rel1, relabel_nodes=True, num_nodes=max(self.clus_src[idx].max()+1, data.edge_index1.max()+1)) \
                                    if len(self.clus_src[idx]) > 0 else (torch.tensor([[]]), torch.tensor([]))
        data.edge_index2, data.rel2 = subgraph(subset=self.clus_tgt[idx],
                                               edge_index=data.edge_index2,
                                               edge_attr=data.rel2, 
                                               relabel_nodes=True, num_nodes=max(self.clus_tgt[idx].max()+1, data.edge_index2.max()+1)) \
                                    if len(self.clus_tgt[idx]) > 0 else (torch.tensor([[]]), torch.tensor([]))

        
        data.num_node1 = data.x1.size(0)
        data.num_node2 = data.x2.size(0)
        data.num_edge1 = data.edge_index1.size(1)
        data.num_edge2 = data.edge_index2.size(1)
        data.map1 = self.src_map[idx] if self.src_map is not None else torch.arange(self.clus_src[idx].size(0))
        data.map2 = self.tgt_map[idx] if self.tgt_map is not None else torch.arange(self.clus_tgt[idx].size(0))
        data.assoc1 = self.clus_src[idx]
        data.assoc2 = self.clus_tgt[idx]
        data.label = self.__compute_label__(self.clus_src[idx], self.clus_tgt[idx])

        return data
    
    def __len__(self):
        return len(self.clus_src)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f')')
        


       
class ClusterLoader(torch.utils.data.DataLoader):
    r"""
    Args:
        cluster_data (ClusterData): The already
            partioned data object.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super().__init__(range(len(cluster_data)), collate_fn=self.__collate__,
                         **kwargs)
    
    def __collate__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.cluster_data[batch]
        
        
            
            