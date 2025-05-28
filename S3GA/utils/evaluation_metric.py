import torch

from torch_geometric.utils import k_hop_subgraph
def hit_at_1_per_batch(pmat_pred, test_y, size=None):
    """
    test_y: groudtruth per batch.
    """
    if size is None:
        size = pmat_pred.size()
    per_label = torch.sparse_coo_tensor(indices=test_y, values=torch.ones(test_y.shape[1]), size=size, device=pmat_pred.device)
    if not isinstance(pmat_pred, torch.sparse.FloatTensor):
        pmat_pred = torch.sparse_coo_tensor(indices=pmat_pred, values=torch.ones(pmat_pred.shape[1]), size=size, device=pmat_pred.device)
    num_acc = torch.sparse.sum(pmat_pred.mul(per_label))
    return num_acc
    
def hit_at_1_sparse_batch(batch_id, pmat_pred, test_y, num_src, num_tgt, cluster_id_s=None, cluster_id_t=None, indices_s=None, indices_t=None):
    """hit_at_1 metric with batch size

    Args:
        batch_id (int): cluster_id
        pmat_pred (sparse_coo_tensor): [2, min(num_src_nodes, num_tgt_nodes)]
        test_y (tensor): [2, num_test_nodes], ground truth.
        cluster_id_s (tensor): [num_row, ], source KG's node clustering
        cluster_id_t (tensor): [num_col, ], target KG's node clustering
    """
    # re-code pmat_pred
    if indices_s is None:
        indices_s = torch.nonzero(cluster_id_s == batch_id).squeeze()
    row = indices_s[pmat_pred._indices()[0]]
    
    if indices_t is None:
        indices_t = torch.nonzero(cluster_id_t == batch_id).squeeze()
    col = indices_t[pmat_pred._indices()[1]]
    
    pmat_pred_sparse = torch.sparse_coo_tensor(indices=torch.stack((row, col),dim=0), values=torch.ones_like(row).to(pmat_pred.device), size=[num_src, num_tgt])
    y = torch.sparse_coo_tensor(indices=test_y, values=torch.ones_like(test_y[0]).to(test_y.device), size=[num_src, num_tgt])
    # acc = torch.sparse.sum(pmat_pred_sparse * y) / torch.tensor(test_y.size(1)).to(test_y.device)
    acc = torch.sparse.sum(pmat_pred_sparse * y)
    # from utils.config import cfg 
    # acc = torch.sparse.sum(pmat_pred_sparse * y) / torch.div(test_y.size(1), cfg.NUM_CENTROIDS, rounding_mode='floor').to(test_y.device)
    
    
    return acc

def hit_at_10_sparse_batch(batch_id, pred_10, test_y, 
                           cluster_id_s=None, cluster_id_t=None, 
                           indices_s=None, indices_t=None, 
                           map_s=None, map_t=None):
    
    if indices_s is None:
        indices_s = torch.nonzero(cluster_id_s == batch_id).squeeze()
    row = indices_s[map_s]
    # bias = torch.where(row.unsqueeze(1) == test_y[0].unsqueeze(0))[1]
    test_map = {value.item(): idx for idx, value in enumerate(test_y[0])}
    bias = torch.tensor([test_map.get(value.item(), -1) for value in row])

    col_test_y = test_y[1][bias]

    if indices_t is None:
        indices_t = torch.nonzero(cluster_id_t == batch_id).squeeze()
    
    col_pred = indices_t[pred_10] # original_indices of sim

    hit10 = (col_pred == col_test_y.unsqueeze(1)).sum()
    return hit10
        

def hit_at_cluster(test_y, cluster_s, cluster_t):
    """evaluate the performance of clustering.

    Args:
        test_y (tensor): [2, num_test_nodes], ground truth.
        cluster_s (tensor): [num_row, ], source KG's node clustering
        cluster_t (tensor): [num_col, ], target KG's node clustering
    """
    cluster_s = cluster_s.to(test_y.device)
    cluster_t = cluster_t.to(test_y.device)
    test_src_cluster = cluster_s[test_y[0]]
    test_tgt_cluster = cluster_t[test_y[1]]
    acc = torch.sum(test_src_cluster == test_tgt_cluster) / torch.tensor(test_y.shape[1])
    acc_idx = torch.where(test_src_cluster == test_tgt_cluster)[0]
    src_acc_idx = test_y[0][acc_idx]
    tgt_acc_idx = test_y[1][acc_idx]
    return acc, (src_acc_idx, tgt_acc_idx)

def hit_at_batch_cluster(test_y, clus_src, clus_tgt, node_mapping_s=None, node_mapping_t=None):
    """
    Args:
    test_y (torch.tensor): [2, num_test_nodes]. Ground truth.
    clus_src (list): [torch.tensor, torch.tensor, ]. clus_src[i] means the node idx belong to batch i.
    node_mapping_s (Optional, list): [torch.tensor, torch.tensor, ]. node_mapping_s 
    """ 
    num = 0
    num_nodes = 0
    srcs_acc_ori, tgts_acc_ori = [], []
    srcs_acc_all, tgts_acc_all = [], []
    if node_mapping_s is not None and node_mapping_t is not None:
        for i in range(len(clus_src)):
            clus_src[i] = clus_src[i].to(test_y.device)
            clus_tgt[i] = clus_tgt[i].to(test_y.device)
            tmp_src = torch.isin(test_y[0], clus_src[i])
            tmp_tgt = torch.isin(test_y[1], clus_tgt[i])
            indices = torch.logical_and(tmp_src, tmp_tgt)
            # print(torch.any(clus_src[i] == 99524), torch.any(clus_tgt[i] == 8))
            if indices.size(0) > 0:
                src_idx = (test_y[0][indices].unsqueeze(1) == clus_src[i][node_mapping_s[i]]).nonzero(as_tuple=True)[1]
                tgt_idx = (test_y[1][indices].unsqueeze(1) == clus_tgt[i][node_mapping_t[i]]).nonzero(as_tuple=True)[1]
                srcs_acc_ori.append(node_mapping_s[i][src_idx])
                tgts_acc_ori.append(node_mapping_t[i][tgt_idx])
                
                srcs_acc_all.append(
                    (test_y[0][indices].unsqueeze(1) == clus_src[i]).nonzero(as_tuple=True)[1])
                tgts_acc_all.append(
                    (test_y[1][indices].unsqueeze(1) == clus_tgt[i]).nonzero(as_tuple=True)[1])
            else:
                srcs_acc_ori.append(torch.tensor([]))
                tgts_acc_ori.append(torch.tensor([]))
                srcs_acc_all.append(torch.tensor([]))
                tgts_acc_all.append(torch.tensor([]))
            
            # tmp = torch.isin(test_y[0], clus_src[i][node_mapping_s[i]])
            # num += torch.isin(clus_tgt[i][node_mapping_t[i]], test_y[1][tmp]).sum()
            tmp = torch.isin(test_y[0], clus_src[i])
            num += torch.isin(clus_tgt[i], test_y[1][tmp]).sum()
    else:
        for i in range(len(clus_src)):
            num_nodes += clus_src[i].shape[0]
            clus_src[i] = clus_src[i].to(test_y.device)
            clus_tgt[i] = clus_tgt[i].to(test_y.device)
            tmp = torch.isin(test_y[0], clus_src[i])
            clu_acc = torch.isin(clus_tgt[i], test_y[1][tmp]).sum()
            num += clu_acc
            # print("clu_acc\t", i, ":\t", clu_acc, clu_acc/ tmp.sum())
    return num / test_y.shape[1], (srcs_acc_ori, tgts_acc_ori), (srcs_acc_all, tgts_acc_all), num / num_nodes

def hit_at_part_ori_vs_neig(test_y, clus_src, clus_tgt,
                            node_mapping_s, node_mapping_t):
    """
    xxxxxooo
    xxxxxooo
    xxxxxooo
    mmmmmnnn
    mmmmmnnn
    calculating partition acc of xxx and ooo and mmm.
    """
    num = 0
    src_acc, tgt_acc =0.0, 0.0
    neigs_acc = torch.full((test_y.shape[1], ), 0)
    for i in range(len(clus_src)):
        if clus_src[i].size(0) == 0 or clus_tgt[i].size(0) == 0:
            continue
        clus_src[i] = clus_src[i].to(test_y.device)
        clus_tgt[i] = clus_tgt[i].to(test_y.device)
        ori_src = torch.isin(test_y[0], clus_src[i][node_mapping_s[i]])
        num += torch.isin(test_y[1][ori_src], clus_tgt[i]).sum()

        neig_mask = ~torch.isin(torch.arange(len(clus_src[i])), node_mapping_s[i])
        neig_src = torch.isin(test_y[0], clus_src[i][neig_mask])
        # ## 测试用
        # tgt_num = torch.isin(test_y[1][neig_src], clus_tgt[i][node_mapping_t[i]]).sum()
        # tgt_acc += tgt_num/neig_mask.sum() if neig_mask.sum() !=0 else tgt_num

        num += torch.isin(test_y[1][neig_src], clus_tgt[i][node_mapping_t[i]]).sum()
        neig_mask_ = ~torch.isin(torch.arange(len(clus_tgt[i])), node_mapping_t[i])
        neig_tgt = torch.isin(test_y[1], clus_tgt[i][neig_mask_])
        # ## 测试用
        # src_num = torch.isin(test_y[0][neig_tgt], clus_src[i][node_mapping_s[i]]).sum()
        # src_acc += src_num/neig_mask_.sum() if neig_mask_.sum() !=0 else src_num

        acc_idx = torch.logical_and(neig_src, neig_tgt)
        neigs_acc[acc_idx] += 1
        # print(clus_src[i].shape[0], clus_tgt[i].shape[0], node_mapping_s[i].shape[0], node_mapping_t[i].shape[0], src_acc, tgt_acc)
    return num / test_y.shape[1], neigs_acc

'''
    identify the inaccurate nodes
'''
def inaccurate_nodes(test_y, clus_src, clus_tgt, 
                     edge_index_src, edge_index_tgt):
    """
    Args:
    edge_index_src: original large graph's edge_index (src).
    edge_index_tgt: original large graph's edge_index (tgt).
    """
    for i in range(len(clus_src)):
        clus_src[i] = clus_src[i].to(test_y.device)
        clus_tgt[i] = clus_tgt[i].to(test_y.device)
        tmp_src = torch.isin(test_y[0], clus_src[i])
        inacc_src = ~torch.isin(test_y[1][tmp_src], clus_tgt[i])
        inacc_src_node = test_y[1][tmp_src][inacc_src] # 希望 target graph 中被选中的点
        
        tmp_tgt = torch.isin(test_y[1], clus_tgt[i])
        inacc_tgt = ~torch.isin(test_y[0][tmp_tgt], clus_src[i])
        inacc_tgt_node = test_y[0][tmp_tgt][inacc_tgt] # 希望 source graph 中被选中的点
        
        # 看看 n-hop 中的占据比例
        for h in range(1, 3):
            src_h_sets, _, _, _ = k_hop_subgraph(node_idx=clus_src[i], 
                                        num_hops=h, edge_index=edge_index_src,
                                        num_nodes=max(clus_src[i].max(), edge_index_src.max())+1,)
            src_ratio = torch.isin(src_h_sets, inacc_src_node).sum() / inacc_src_node.shape[0]
            print("src_clus_id:", i, f"\t hop{h}:", "\t ratio/num_neighbors:", src_ratio)

            tgt_h_sets, _, _, _ = k_hop_subgraph(node_idx=clus_tgt[i],
                                        num_hops=h, edge_index=edge_index_tgt,
                                        num_nodes=max(clus_tgt[i].max(), edge_index_tgt.max())+1,)
            tgt_ratio = torch.isin(tgt_h_sets, inacc_tgt_node).sum() / inacc_tgt_node.shape[0]
            print("tgt_clus_id:", i, f"\t hop{h}:", "\t ratio/num_neighbors:", tgt_ratio)
        
    