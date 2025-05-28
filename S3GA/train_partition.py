
import os
import os.path as osp
from pathlib import Path
from datetime import datetime
import time

import xlwt
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_sparse.tensor import SparseTensor

# from torch_geometric.utils import to_dense_batch
from torch.cuda.amp import autocast as autocast


from yaml import parse
from GSSL.clustering import *
from data.mkdata_CEA import mkdata
from utils.data_clustering import clustering, add_neighbor
from utils.model_sl import load_model, save_model
from utils.evaluation_metric import hit_at_1_sparse_batch, hit_at_batch_cluster, hit_at_part_ori_vs_neig
# from utils.kmeans import KMeans
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.print_easydict import print_easydict
from utils.parse_args import parse_args
from utils.visualization import construct_graph_
# from utils.histogram import histogram_degree
from dataset.data_loader import ClusterData, EvalClusterData, ClusterLoader

# from utils.dataloader import LMCLoader, EvalSubgraphLoader
# from utils.metis import metis_pair, permute_pair, gt_subgraph
from utils.config import cfg

def train_eval(model, optimizer, train_loader, eval_loader, tfboardwriter, num_src, num_tgt, start_epoch=0, num_epochs=50):
    print("Start training alignment model...")
    
    evice = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH, pair) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True) # parents= True, 创建这个路径的任何缺失的父目录

    #  断点恢复
    model_path, optim_path = '', ''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    
    if len(model_path) > 0:
        print('Loading model from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))


    for e in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(e, num_epochs - 1))
        print('-' * 10)
        # --- train --- #
        model.train()
        tot_loss = 0
        running_since = time.time()
        iter = 0
        for graph in train_loader:
            graph = graph.to(device)
            optimizer.zero_grad()
            if graph.x1.size(0) == 0 or graph.x2.size(0) == 0:
                continue
            
            _, pred_m, perm_sparse, mask, _ = model(graph.x1, graph.edge_index1, graph.rel1, None,
                                                    graph.x2, graph.edge_index2, graph.rel2, None, 
                                                    graph.map1, graph.map2)
            if pred_m.size(0) >= pred_m.size(1):
                # loss = F.cross_entropy(pred_m.T[mask], perm_sparse._indices()[0], reduction='sum') / torch.tensor(pred_m.shape[1])
                loss = F.cross_entropy(pred_m.T, perm_sparse._indices()[0], reduction='sum') / torch.tensor(pred_m.shape[1])
            else:
                # loss = F.cross_entropy(pred_m[mask], perm_sparse._indices()[1], reduction='sum') / torch.tensor(pred_m.shape[0])
                loss = F.cross_entropy(pred_m, perm_sparse._indices()[1], reduction='sum') / torch.tensor(pred_m.shape[0])
            
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            iter += 1

        running_speed = time.time() - running_since
        print('Epoch {:<4} train:{:>4.2f}s Loss={:<8.4f}'.format(e, running_speed, tot_loss))
            
        # --- save model --- #
        if e % cfg.STATISTIC_STEP == 0:
            loss_dict = dict()
            loss_dict['loss_perm'] = tot_loss
            tfboardwriter.add_scalars('loss', loss_dict, e)
            save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(e + 1)))
            torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(e + 1)))

            model = evaluate(model=model, eval_loader=eval_loader, tfboardwriter=tfboardwriter,
                            num_src=num_src, num_tgt=num_tgt)
    print(f"Finish {num_epochs} epochs training.")
    return model

@torch.no_grad()
def evaluate(model, eval_loader, tfboardwriter, num_src, num_tgt):
    print("Start evaluating alignment model...")
    num_clu, hits1 = 0, 0
    model.eval()
    eval_since = time.time()
    for graph in eval_loader:
        graph = graph.to(device)
        if graph.x1.size(0) == 0 or graph.x2.size(0) == 0:
            num_clu += 1
            continue
        _, _, _, _, perm_sparse = model(graph.x1, graph.edge_index1, graph.rel1, None, 
                                                        graph.x2, graph.edge_index2, graph.rel2, None, 
                                                        graph.map1, graph.map2)
        hist1_noneigh = hit_at_1_sparse_batch(batch_id=num_clu, pmat_pred=perm_sparse, test_y=graph.gt_y, 
                                              num_src=num_src, num_tgt=num_tgt, indices_s=graph.assoc1, 
                                              indices_t=graph.assoc2)
        hits1 += hist1_noneigh

        # hits1_neigh = hit_at_1_sparse_batch(batch_id=num_clu, pmat_pred=perm_sparse_neigh, test_y=graph.gt_y,
        #                                     num_src=num_src, num_tgt=num_tgt, indices_s=graph.assoc1, 
        #                                     indices_t=graph.assoc2)
        hits1_dict = dict()
        # hits1_dict['hist1_neigh'] = hits1_neigh.item()
        hits1_dict['hits1_noneigh'] = hist1_noneigh.item()
        tfboardwriter.add_scalars('his1_per_clu', hits1_dict, num_clu)
        num_clu += 1
    hits1 = hits1 / torch.tensor(graph.gt_y.size(1)).to(device)
    print(f'Hits@1: {hits1:.4f}')
    eval_speed = time.time() - eval_since
    print(f"Finish evaluating in {eval_speed}s.")
    return model

@torch.no_grad()
def re_clustering(model, eval_loader, device='cpu'):
    print("Start re-clustering...")
    model.eval()
    c_cens, non_embs_x1, non_embs_x2, clus_src, clus_tgt = [], [], [], [], []
    non_allocs1, non_allocs2 = torch.tensor([], dtype=torch.int32, device=device), torch.tensor([], dtype=torch.int32, device=device)
    
    re_clus_since = time.time()

    for graph in eval_loader:
        graph = graph.to(device)
        if graph.x1.size(0) > 0 and graph.x2.size(0) > 0:
            c_cen, alloc1, alloc2, non_alloc1, non_alloc2, non_emb_x1, non_emb_x2 = model.embed(graph.x1, graph.edge_index1, graph.rel1, None,
                                                                            graph.x2, graph.edge_index2, graph.rel2, None, 
                                                                            graph.map1, graph.map2)
            if len(c_cen) != 0:
                c_cens.append(torch.cat(c_cen, dim=0))
                clus_src.append(graph.assoc1[alloc1])
                clus_tgt.append(graph.assoc2[alloc2])
            non_allocs1 = torch.cat([non_allocs1, graph.assoc1[non_alloc1]], dim=0)
            non_allocs2 = torch.cat([non_allocs2, graph.assoc2[non_alloc2]], dim=0)
            non_embs_x1.append(torch.cat(non_emb_x1, dim=1))
            non_embs_x2.append(torch.cat(non_emb_x2, dim=1))
            # # visualization
            # construct_graph_(node_index=graph.assoc1[alloc1], edge_index=graph.edge_index1, output=f'{len(c_cens)}_src')
            # construct_graph_(node_index=graph.assoc2[alloc2], edge_index=graph.edge_index2, output=f'{len(c_cens)}_tgt')

        elif graph.x1.size(0) > 0: # graph.x2.size(0) == 0
            non_allocs1 = torch.cat([non_allocs1, graph.assoc1], dim=0)
            non_emb_x1 = model.encoder(graph.x1, graph.edge_index1)
            non_embs_x1.append(torch.cat(non_emb_x1, dim=1))
        elif graph.x2.size(0) > 0: 
            non_allocs2 = torch.cat([non_allocs2, graph.assoc2], dim=0)
            non_emb_x2 = model.encoder(graph.x2, graph.edge_index2)
            non_embs_x2.append(torch.cat(non_emb_x2, dim=1))
        
    c_cens = torch.stack(c_cens, dim=0)
    non_embs_x1 = torch.cat(non_embs_x1, dim=0)
    non_embs_x2 = torch.cat(non_embs_x2, dim=0)
    assert non_allocs1.size(0) == non_embs_x1.size(0), 'Re-clustering: src id and emb are not match!'
    assert non_allocs2.size(0) == non_embs_x2.size(0), 'Re-clustering: tgt id and emb are not match!'
    re_clus_speed = time.time() - re_clus_since
    print(f'find that non_alloc_src: {non_allocs1.size(0)}, non_alloc_tgt: {non_allocs2.size(0)} with eval speed {re_clus_speed}s')

    clu_src_id = torch.einsum("am, cm->ac", non_embs_x1[:, -cfg.MODEL.HIDDEN_CHANNEL:], c_cens[:, -cfg.MODEL.HIDDEN_CHANNEL:])
    clu_tgt_id = torch.einsum("am, cm->ac", non_embs_x2[:, -cfg.MODEL.HIDDEN_CHANNEL:], c_cens[:, -cfg.MODEL.HIDDEN_CHANNEL:])
    for i in range(1, cfg.MODEL.NUM_LAYER):
        clu_src_id += torch.einsum("am,cm->ac", non_embs_x1[:, -cfg.MODEL.HIDDEN_CHANNEL * (i+1) :-cfg.MODEL.HIDDEN_CHANNEL * i], c_cens[:, -cfg.MODEL.HIDDEN_CHANNEL * (i+1) :-cfg.MODEL.HIDDEN_CHANNEL * i])
        clu_tgt_id += torch.einsum("am,cm->ac", non_embs_x2[:, -cfg.MODEL.HIDDEN_CHANNEL * (i+1) :-cfg.MODEL.HIDDEN_CHANNEL * i], c_cens[:, -cfg.MODEL.HIDDEN_CHANNEL * (i+1) :-cfg.MODEL.HIDDEN_CHANNEL * i])
    
    clu_src_id += torch.einsum("am, cm->ac", non_embs_x1[:, :-cfg.MODEL.HIDDEN_CHANNEL * cfg.MODEL.NUM_LAYER], c_cens[:, :-cfg.MODEL.HIDDEN_CHANNEL * cfg.MODEL.NUM_LAYER])
    clu_tgt_id += torch.einsum("am, cm->ac", non_embs_x2[:, :-cfg.MODEL.HIDDEN_CHANNEL * cfg.MODEL.NUM_LAYER], c_cens[:, :-cfg.MODEL.HIDDEN_CHANNEL * cfg.MODEL.NUM_LAYER])
    
    non_alloc_src = torch.argmax(clu_src_id, dim=-1)
    non_alloc_tgt = torch.argmax(clu_tgt_id, dim=-1)
    re_clus_speed = time.time() - re_clus_since
    print(f"re-clustering speed: {re_clus_speed}")
    for c in range(c_cens.size(0)):
        idx1 = torch.nonzero(non_alloc_src == c).squeeze(1)
        clus_src[c] = torch.cat([clus_src[c], non_allocs1[idx1]], dim=0)
        idx2 = torch.nonzero(non_alloc_tgt == c).squeeze(1)
        clus_tgt[c] = torch.cat([clus_tgt[c], non_allocs2[idx2]], dim=0) 
    print("Finish Re-clustering! ")    
    return clus_src, clus_tgt

if __name__ == '__main__':
    args = parse_args('Entity clustering then Alignment')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    datasets = eval(cfg.DATASET_FULL_NAME)

    for pair in ['de']:
        print('*' * 20)
        print(pair)
        print('*' * 20)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        align_model = alignment_with_neighbor_nomask().to(device)
        # align_model = alignment_use_neighbor().to(device)
        align_optim = optim.Adam(align_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.LR_DECAY)
        
        if not Path(cfg.OUTPUT_PATH, pair).exists():
            Path(cfg.OUTPUT_PATH, pair).mkdir(parents=True)

        now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH, pair) / 'tensorboard' / 'training_{}'.format(now_time)))
        
        file_path = Path(cfg.OUTPUT_PATH, pair, 'cluster_lists.pth')

        with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH, pair) / ('train_log_' + now_time + '.log'))) as _:
            print_easydict(cfg)
            # --- load data --- #
            print("Loading data ... ")
            dataset = datasets(cfg.DATASET_PATH, pair, emb_model='Labse', outlier=False)[0]
            # -- load quickly clustering results --- #
            if file_path.exists():
                clus_list = torch.load(file_path)
                clus_src, clus_tgt = clus_list['cluster_src'], clus_list['cluster_tgt']
            else:
                clus_src, clus_tgt, (srcs_acc, tgts_acc) = clustering(dataset.x1, dataset.x2, 
                                                    num_parts=cfg.NUM_CENTROIDS,
                                                    gt_y=dataset.gt_y,
                                                    clu_method='K-means', 
                                                    # clu_method='Metis',
                                                    # clu_method='groundtruth',
                                                    edge_index1=dataset.edge_index1, 
                                                    edge_index2=dataset.edge_index2,
                                                    edge_rel1=dataset.rel1, 
                                                    edge_rel2=dataset.rel2)
                # --- quickly re-clustering --- #
                for iter in range(cfg.TRAIN.NUM_ITER):
                    # --- add the neighors for each cluster --- #
                    clus_src_neig, clus_tgt_neig, src_map, tgt_map, src_edge, tgt_edge = add_neighbor(num_parts=len(clus_src),
                                                                        clus_src=clus_src, clus_tgt=clus_tgt, 
                                                                        edge_index1=dataset.edge_index1,
                                                                        edge_index2=dataset.edge_index2,
                                                                        neighbor_method='global_random', 
                                                                        num_src_nodes=dataset.x1.size(0), num_tgt_nodes=dataset.x2.size(0))
                
                    train_data = ClusterData(data=dataset, num_parts=len(clus_src_neig), cluster_list=(clus_src_neig, clus_tgt_neig),
                                    map_list=(src_map, tgt_map))
                    train_loader = ClusterLoader(train_data, batch_size=1,)
                    print("Finish loading train dataset.")
                    
                    # ---re-cluster --- #
                    clus_src, clus_tgt = re_clustering(model=align_model, eval_loader=train_loader, device=device)
                    print("STEP1:\tHIT_AT_CLUSTER_AFTER_RE_CLUSTERING:", hit_at_batch_cluster(test_y=dataset.gt_y, 
                                                                                    clus_src=clus_src, clus_tgt=clus_tgt))
                # save the clustering results.
                
                torch.save({'cluster_src': clus_src, 'cluster_tgt': clus_tgt}, file_path)
                print(f"clustering results saved in {file_path}")

            # print("HIT_AT_CLUSTER:", hit_at_batch_cluster(test_y=dataset.gt_y, clus_src=clus_src, clus_tgt=clus_tgt))
            # --- select 1-hop neighbors to train GNN --- #
            for iter_  in range(cfg.TRAIN.NUM_ITER):
                # --- eval --- # (epoch 0)
                eval_data = ClusterData(data=dataset,num_parts=len(clus_src), cluster_list=(clus_src, clus_tgt),)
                eval_loader = ClusterLoader(eval_data, batch_size=1,)
                print("Finish loading eval dataset.")
                align_model = evaluate(model=align_model, eval_loader=eval_loader, tfboardwriter=tfboardwriter,
                             num_src=dataset.x1.size(0), num_tgt=dataset.x2.size(0))
                # --- add neighbors --- #
                clus_src_neig, clus_tgt_neig, src_map, tgt_map, src_edge, tgt_edge = add_neighbor(num_parts=len(clus_src),
                                                                    clus_src=clus_src, clus_tgt=clus_tgt, 
                                                                    edge_index1=dataset.edge_index1,
                                                                    edge_index2=dataset.edge_index2,
                                                                    neighbor_method='path_more', 
                                                                    num_src_nodes=dataset.x1.size(0), num_tgt_nodes=dataset.x2.size(0))
                ori_vs_neig_acc, neig_vs_neig = hit_at_part_ori_vs_neig(dataset.gt_y, clus_src_neig, clus_tgt_neig, 
                                        node_mapping_s=src_map, node_mapping_t=tgt_map)
                
                print("HIT_AT_PARTITION_AFTER_ADD_NEIGHBOR_PATH_MORE:", ori_vs_neig_acc)
                print(torch.topk(neig_vs_neig, k=20, ))

                # # --- visualization --- #
                # for i in range(len(clus_src_neig)):
                #     if len(clus_src_neig[i]) != 0 and len(clus_tgt_neig[i]) != 0:
                #         construct_graph_(node_index=clus_src_neig[i], edge_index=dataset.edge_index1, output=f'{iter}_{i}_src')
                #         construct_graph_(node_index=clus_tgt_neig[i], edge_index=dataset.edge_index2, output=f'{iter}_{i}_tgt')
                # exit()
                #  --- train --- #
                train_data = ClusterData(data=dataset, num_parts=len(clus_src_neig), cluster_list=(clus_src_neig, clus_tgt_neig),
                                map_list=(src_map, tgt_map))
                train_loader = ClusterLoader(train_data, batch_size=1,)
                print("Finish loading train dataset with neighbors.")

                # # --- eval --- # (after training)
                # eval_data = EvalClusterData(data=dataset, num_parts=len(clus_src), cluster_list=(clus_src, clus_tgt), 
                #                             map_list=(src_map, tgt_map))
                # eval_loader = ClusterLoader(eval_data, batch_size=1,)
                # print("Finish loading eval dataset.")
                align_model = train_eval(model=align_model, optimizer=align_optim, train_loader=train_loader, eval_loader=train_loader,
                              tfboardwriter=tfboardwriter, num_src=dataset.x1.size(0), num_tgt=dataset.x2.size(0), start_epoch=0, num_epochs=cfg.TRAIN.NUM_EPOCHS)
                
                # ---re-cluster --- #
                clus_src, clus_tgt = re_clustering(model=align_model, eval_loader=train_loader, device=device)
                print("STEP2:\tHIT_AT_CLUSTER_AFTER_RE_CLUSTERING:", hit_at_batch_cluster(test_y=dataset.gt_y, 
                                                                                  clus_src=clus_src, clus_tgt=clus_tgt))

            print("Done.")