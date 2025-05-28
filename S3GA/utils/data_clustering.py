import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, coalesce
# from torch_geometric.utils import subgraph, k_hop_subgraph
from utils.k_hop_subgraph_sampler import k_hop_subgraph, k_hop_subgraph_path, k_hop_subgraph_edge_more, k_hop_subgraph_edge_more_matrix
# from utils.kmeans import KMeans
from utils.metis import Metis
# from utils.add_neighbors import GraphSaintRandomWalk, GraphSaintEdge
from utils.evaluation_metric import hit_at_cluster, hit_at_batch_cluster
from typing import Optional

import os.path as osp

import copy

def clustering(src_emb, tgt_emb, num_parts, gt_y, 
               clu_method='K-means', 
               edge_index1: Optional[torch.Tensor] = None, 
               edge_index2: Optional[torch.Tensor] = None,
               edge_rel1: Optional[torch.Tensor] = None, 
               edge_rel2: Optional[torch.Tensor] = None, 
               gcn_model=None, save_dir = None,
               ):
    filename = f'partition_{num_parts}{clu_method}.pt'
    path = osp.join(save_dir or '', filename)
    if save_dir is not None and osp.exists(path):
        clusters = torch.load(path)
    else:
        if clu_method.lower() == 'k-means':
            # --· faiss-kmeans · -- #
            import faiss
            import time
            clu_since = time.time()
            cluster = faiss.Kmeans(d=src_emb.size(1), k=num_parts, gpu=True)
            X = torch.cat([src_emb, tgt_emb], dim=0).detach().cpu().numpy()
            cluster.train(X)
            _, clusters = cluster.assign(X)
            clu_time = time.time() - clu_since
            print("*" * 25)
            print(f"k-means time cost: {clu_time}s")
            clusters = torch.tensor(clusters)
            # # --· sklearn-kmeans ·-- # 
            # from sklearn.cluster import KMeans
            # import numpy as np
            # X = torch.cat([src_emb, tgt_emb], dim=0).detach().cpu().numpy()
            # cluster = KMeans(n_clusters=num_parts, random_state=0, n_init=10).fit(X)
            # clusters = torch.from_numpy(cluster.labels_)

            
            if save_dir is not None:
                torch.save(clusters, path)
            clu_src, clu_tgt = clusters[:src_emb.size(0)], clusters[src_emb.size(0):]
            acc, (src_idx, tgt_idx) = hit_at_cluster(gt_y, clu_src, clu_tgt)
            print("Hit_at_CLUSTER_INITIAL:", acc)   
            clus_src, clus_tgt = [], []
            srcs_acc, tgts_acc = [], []
            for c in range(num_parts):
                sub_idx1 = torch.nonzero(clu_src == c).squeeze(1)
                clus_src.append(sub_idx1)
                sub_idx2 = torch.nonzero(clu_tgt == c).squeeze(1)
                clus_tgt.append(sub_idx2)

                acc_idx1 = torch.nonzero(clu_src[src_idx] == c).squeeze(1)
                # srcs_acc.append(src_idx[acc_idx1])
                src_acc = torch.nonzero(src_idx[acc_idx1].unsqueeze(1) == sub_idx1, as_tuple=True)[1]
                srcs_acc.append(src_acc)
                acc_idx2 = torch.nonzero(clu_tgt[tgt_idx] == c).squeeze(1)
                # tgts_acc.append(tgt_idx[acc_idx2])
                tgt_acc = torch.nonzero(tgt_idx[acc_idx2].unsqueeze(1) == sub_idx2, as_tuple=True)[1]
                tgts_acc.append(tgt_acc)
    
        elif clu_method.lower() == 'metis':
            cluster = Metis(n_clusters=num_parts)
            clu_src = cluster.fit(src_emb, edge_index1, edge_rel1, batch_training=False, nx=True)
            clu_tgt = cluster.predict(tgt_emb, edge_index2, edge_rel2, batch_training=False)
            # clu_src = cluster.fit(tgt_emb, edge_index2, edge_rel2, batch_training=False, nx=True)
            # clu_tgt = cluster.predict(src_emb, edge_index1, edge_rel1, batch_training=False)
            if save_dir is not None:
                torch.save(torch.cat([clu_src, clu_tgt], dim=0))
            acc, (src_idx, tgt_idx) = hit_at_batch_cluster(gt_y, clu_src, clu_tgt)
            print("Hit_at_CLUSTER_INITIAL:", acc)
            clus_src, clus_tgt = [], []
            srcs_acc, tgts_acc = [], []
            for c in range(num_parts):
                sub_idx1 = torch.nonzero(clu_src == c).squeeze(1)
                clus_src.append(sub_idx1)
                sub_idx2 = torch.nonzero(clu_tgt == c).squeeze(1)
                clus_tgt.append(sub_idx2)

                acc_idx1 = torch.nonzero(clu_src[src_idx] == c).squeeze(1)
                srcs_acc.append(src_idx[acc_idx1])
                acc_idx2 = torch.nonzero(clu_tgt[tgt_idx] == c).squeeze(1)
                tgts_acc.append(tgt_idx[acc_idx2])
        elif clu_method.lower() == 'groundtruth':
            clus_src = list(torch.chunk(gt_y[0], chunks=num_parts, dim=0))
            clus_tgt = list(torch.chunk(gt_y[1], chunks=num_parts, dim=0))
            acc, _, _ = hit_at_batch_cluster(gt_y, clus_src, clus_tgt)
            print("Hit_at_CLUSTER_INITIAL:", acc) 
            srcs_acc, tgts_acc = torch.tensor([]), torch.tensor([])
        else:
            raise ValueError('Invalid cluster method')
        
    return clus_src, clus_tgt, (srcs_acc, tgts_acc)
    

def add_neighbor(num_parts, clus_src, clus_tgt, edge_index1, edge_index2, neighbor_method='k_hop',
                 num_src_nodes=None, num_tgt_nodes=None):
    new_clus_src, new_clus_tgt = [[] for _ in range(len(clus_src))], [[] for _ in range(len(clus_tgt))]
    src_map, tgt_map = [], []
    src_edge, tgt_edge = [], []
    if neighbor_method == 'global_random':
        for i in range(num_parts):
            if clus_src[i].size(0) != 0:
                src_nodes = torch.ones(max(edge_index1.max(), clus_src[i].max())+1, 
                                            dtype=torch.bool)
                node_mapping1 = torch.arange(clus_src[i].size(0))
                src_nodes[clus_src[i]] = False
                other_nodes = torch.where(src_nodes)[0]
                indices = torch.randperm(src_nodes.sum())[:int(clus_src[i].size(0)/3)]
                # print('src: ', clus_src[i].shape[0], " --> ", other_nodes[indices].shape[0])
                new_clus_src[i] = torch.cat([clus_src[i], other_nodes[indices]], dim=0)
            else:
                new_clus_src[i], node_mapping1 = torch.tensor([]), torch.tensor([])
            src_map.append(node_mapping1)
            
            if clus_tgt[i].size(0) !=0:
                tgt_nodes = torch.ones(max(edge_index2.max(), clus_tgt[i].max())+1, 
                                     dtype=torch.bool)
                node_mapping2 = torch.arange(clus_tgt[i].size(0))
                tgt_nodes[clus_tgt[i]] = False
                other_nodes = torch.where(tgt_nodes)[0]
                indices = torch.randperm(tgt_nodes.sum())[:int(clus_tgt[i].size(0)/3)]
                # print('tgt: ', clus_tgt[i].shape[0], " --> ", other_nodes[indices].shape[0])
                new_clus_tgt[i] = torch.cat([clus_tgt[i], other_nodes[indices]], dim=0)
            else:
                new_clus_tgt[i], node_mapping2 = torch.tensor([]), torch.tensor([])
            tgt_map.append(node_mapping2)

    elif neighbor_method == 'fix_random':
        src_random = torch.randperm(num_src_nodes)
        tgt_random = torch.randperm(num_tgt_nodes)
        for i in range(num_parts):
            if clus_src[i].size(0) != 0:
                node_mapping1 = torch.arange(clus_src[i].size(0))
                clus_src[i] = torch.cat([clus_src[i], src_random[:int(clus_src[i].size(0)/3)]], dim=0)
            else:
                clus_src[i], node_mapping1 = torch.tensor([]), torch.tensor([])
            src_map.append(node_mapping1)
            if clus_tgt[i].size(0) != 0:
                node_mapping2 = torch.arange(clus_tgt[i].size(0))
                clus_tgt[i] = torch.cat([clus_tgt[i], tgt_random[:int(clus_tgt[i].size(0)/3)]], dim=0)
            else:
                clus_tgt[i], node_mapping2 = torch.tensor([]), torch.tensor([])
            tgt_map.append(node_mapping2)
    elif neighbor_method == 'none':
        for i in range(num_parts):
            new_clus_src[i] = clus_src[i]
            new_clus_tgt[i] = clus_tgt[i]
            node_mapping1 = torch.arange(clus_src[i].size(0))
            node_mapping2 = torch.arange(clus_tgt[i].size(0))
            src_map.append(node_mapping1)
            tgt_map.append(node_mapping2)
    
   
    else: 
        raise ValueError("Cannot understand the neighbor method.")
            
    
    return new_clus_src, new_clus_tgt, src_map, tgt_map, src_edge, tgt_edge
    # return clus_src, clus_tgt, src_map, tgt_map, src_edge, tgt_edge

def add_neighbor_(num_parts, clus_src, clus_tgt, edge_index1, edge_index2,
                  neighbor_method='path_random', num_src_nodes=None, num_tgt_nodes=None):
    new_clus_tgt = [[] for _ in range(len(clus_tgt))]
    tgt_map = []
    # neighbor_method = 'path_random', 只给clus_tgt选neighbors to make sure num_src <= num_tgt
    for i in range(num_parts):
        if clus_src[i].size(0) <= clus_tgt[i].size(0):
            new_clus_tgt[i] = clus_tgt[i]
            tgt_map.append(torch.tensor([]))
        else:
            subset2, _, node_mapping2, _ = k_hop_subgraph(clus_src[i], 1, edge_index1,
                                                            num_nodes=max(clus_src[i].max()+1, 
                                                                          edge_index1.max()+1),
                                                            relabel_nodes=True, neig_select='random', 
                                                            neig_num=(clus_src[i].size(0)-clus_tgt[i].size(0)))
            
            new_clus_tgt[i] = subset2
            tgt_map.append(node_mapping2)
    return clus_src, new_clus_tgt, tgt_map
    