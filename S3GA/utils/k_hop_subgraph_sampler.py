from typing import List, Optional, Union
import random

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes


def k_hop_subgraph(node_idx, num_hops, edge_index, 
                   relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', 
                   neig_select='random', 
                   neig_num=0):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    neig_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        # k-hop-neighbors
        torch.index_select(node_mask, 0, col, out=neig_mask)
        edge_mask[neig_mask] = False

        if neig_select == 'random':
            indices = torch.randperm(col[edge_mask].size(0))[:neig_num]
            subsets.append(col[edge_mask][indices])
        else:
            subsets.append(col[edge_mask])
            

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def k_hop_subgraph_path(node_idx, num_hops, edge_index, relabel_nodes=False,
                        num_nodes=None, flow='source_to_target', 
                        neig_select='none'):
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index
    
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    neig_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    subsets_path = [node_idx]
    k_hop_neigs = []
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

        # k-hop-neighbors
        torch.index_select(node_mask, 0, col, out=neig_mask)
        edge_mask[neig_mask] = False
        k_hop_neigs.append(col[edge_mask])
        neigs, counts = torch.cat(k_hop_neigs).unique(return_counts=True)
        # cut some neighbors
        # if (counts >=2).sum() > int(node_idx.shape[0]/3):
        #     indices = torch.argsort(counts, dim=0, descending=True)
        #     indices = indices[:int(node_idx.shape[0]/3)]
        #     neigs = neigs[indices]
        # else:
        #     neigs = neigs[counts >=2]
        # random neighbors
        if (counts >=2).sum() > int(node_idx.shape[0]/3):
            indices =  torch.randperm(col[edge_mask].size(0))[:int(node_idx.size(0)/3)]
            neigs = col[edge_mask][indices]
        else:
            indices = torch.randperm(col[edge_mask].size(0))[:(counts >=2).sum()]
            neigs = col[edge_mask][indices]
        subsets_path.append(neigs)

    subset_path, inv = torch.cat(subsets_path).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset_path] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset_path] = torch.arange(subset_path.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    
    return subset_path, edge_index, inv, edge_mask

def k_hop_subgraph_edge_more(node_idx, num_hops, edge_index, relabel_nodes=False,
                             num_nodes=None, flow='source_to_target'):
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index
    
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    neig_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    subsets_path = [node_idx]
    k_hop_neigs = []
    neighs = []
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

        # only k-hop-neighbors
        torch.index_select(node_mask, 0, col, out=neig_mask)
        edge_mask[neig_mask] = False
        k_hop_neigs.append(col[edge_mask])
        neigs, inverse, counts = torch.cat(k_hop_neigs).unique(return_counts=True, return_inverse=True)
        # cut some neighbors
        if (counts >=2).sum() > int(node_idx.shape[0]/3):
            while len(neighs) < int(node_idx.shape[0]/3):
                indices = torch.argmax(counts, dim=-1)
                neighs.append(neigs[indices].item())
                inclu = torch.where(inverse == indices)[0]
                rows = torch.isin(row, row[edge_mask][inclu])
                edge_mask[rows] = False
                if edge_mask.sum() == 0:
                    break
                neigs, inverse, counts = torch.cat([col[edge_mask]]).unique(return_counts=True,return_inverse=True)
            # if int(node_idx.shape[0]/3) - len(neighs) > 0:
            #     neigs, inverse, counts = torch.cat(k_hop_neigs).unique(return_counts=True, return_inverse=True)
            #     indices = torch.argsort(counts)
            #     num_ = torch.isin(neigs[indices][:int(node_idx.shape[0]/3)], neigs).sum()
            #     neighs += neigs[indices][:int(node_idx.shape[0]/3)+num_].cpu().tolist()
                
            neigs = torch.tensor(neighs)
        else:
            neigs = neigs[counts >= 2]
        subsets_path.append(neigs)
    subset_path, inv = torch.cat(subsets_path).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset_path] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset_path] = torch.arange(subset_path.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    
    return subset_path, edge_index, inv, edge_mask

def k_hop_subgraph_edge_more_matrix(node_idx, num_hops, edge_index, relabel_nodes=False,
                                    num_nodes=None, flow='source_to_target'):
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index
    
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    neig_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    subsets_path = [node_idx]
    k_hop_neigs = []
    neighs = []
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

        # only k-hop-neighbors
        torch.index_select(node_mask, 0, col, out=neig_mask)
        edge_mask[neig_mask] = False
        k_hop_neigs.append(col[edge_mask])
        # neigs, inverse, counts = torch.cat(k_hop_neigs).unique(return_counts=True, return_inverse=True)  
        
        adj = SparseTensor(row=row[edge_mask], col=col[edge_mask],
                           value=torch.ones(edge_mask.sum()).to(node_idx.device),)
        if (adj.sum(dim=0) >= 2).sum() > int(node_idx.shape[0]/3):
            while len(neighs) < int(node_idx.shape[0]/3):
                neig = torch.argmax(adj.sum(dim=0))
                neighs.append(neig.item())
                rows = adj.index_select(dim=1, idx=neig.unsqueeze_(0)).storage.row()
                edge_mask[torch.isin(row, rows)] = False
                if not edge_mask.any():
                    break
                adj = SparseTensor(row=row[edge_mask], col=col[edge_mask],
                                   value = torch.ones(edge_mask.sum()).to(node_idx.device),)
            neigs = torch.tensor(neighs)
        
        else:
            neigs = torch.where(adj.sum(dim=0) >= 2)[0]
            
        subsets_path.append(neigs)
        # # --- ·random ·--- #
        # hop_1_neigs = torch.unique(k_hop_neigs[0]) 
        # random_neigs =  torch.randperm(hop_1_neigs.size(0))[:neigs.shape[0]]
        # print("neigs:", torch.sort(random_neigs))
        # print("number neigs:", neigs.shape[0])
        # subsets_path.append(random_neigs)
        
    subset_path, inv = torch.cat(subsets_path).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset_path] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset_path] = torch.arange(subset_path.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    
    return subset_path, edge_index, inv, edge_mask




if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6],[2, 2, 4, 4, 6, 6, 4]])
    # subset, edge_index, mapping, edge_mask = k_hop_subgraph(4, 2, edge_index, relabel_nodes=True)
    subset1, edge_index1, mapping1, edge_mask1 = k_hop_subgraph_edge_more(torch.tensor([4,0]), 1, edge_index, relabel_nodes=True)
    subset2, edge_index2, mapping2, edge_mask2 = k_hop_subgraph_edge_more(torch.tensor([4,0]), 1, edge_index, relabel_nodes=True)
    print(subset1, subset2,)