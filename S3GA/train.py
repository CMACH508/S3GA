
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
from data.openea_CEA import openeav2
from data.DBP15K_slotalign import DBP15K
from utils.data_clustering import clustering, add_neighbor_
from utils.model_sl import load_model, save_model
from utils.evaluation_metric import hit_at_1_sparse_batch, hit_at_10_sparse_batch
# from utils.kmeans import KMeans
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.print_easydict import print_easydict
from utils.parse_args import parse_args
# from utils.visualization import construct_graph_
# from utils.histogram import histogram_degree
from dataset.data_loader import ClusterData, EvalClusterData, SuperClusterData, ClusterLoader

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
    
    model = evaluate(model, eval_loader, tfboardwriter, num_src, num_tgt)
    
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
            if graph.x1.size(0) == 0 or graph.x2.size(0) == 0 or graph.label.size(1) == 0: 
                continue
            _, pred_m, perm_sparse, mask, _ = model(graph.x1, graph.edge_index1, graph.rel1, None,
                                                    graph.x2, graph.edge_index2, graph.rel2, None, 
                                                    graph.map1, graph.map2)
            if pred_m.size(0) >= pred_m.size(1):
                loss = F.cross_entropy(pred_m.T, perm_sparse._indices()[0], reduction='sum') / torch.tensor(pred_m.shape[1])
            else:
                loss = F.cross_entropy(pred_m, perm_sparse._indices()[1], reduction='sum') / torch.tensor(pred_m.shape[0])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            iter += 1
            
        running_speed = time.time() - running_since
        print('Epoch {:<4} train:{:>4.2f}s Loss={:<8.4f}'.format(e, running_speed, tot_loss))
        
        # --- save and evaluate model --- #
        if e % cfg.STATISTIC_STEP == 0:
            loss_dict = dict()
            loss_dict['loss_perm'] = tot_loss
            tfboardwriter.add_scalars('loss', loss_dict, e)
            save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(e + 1)))
            torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(e + 1)))

            # --- evaluate --- #
            model = evaluate(model, eval_loader, tfboardwriter, num_src, num_tgt)

    print(f"Finish {num_epochs} epochs training.")
    return model

@torch.no_grad()
def evaluate(model, eval_loader, tfboardwriter, num_src, num_tgt):
    print("Start evaluating alignment model...")
    num_clu, hits1, hits10 = 0, 0, 0
    non_allocs_src_num, non_allocs_tgt_num = 0, 0
    hits1_inlier, num_inlier = 0, 0
    pseudo_label = []
    model.eval()
    eval_since = time.time()
    for graph in eval_loader:
        graph = graph.to(device)
        if graph.x1.size(0) == 0 or graph.x2.size(0) == 0:
            num_clu += 1
            continue
        _, _, perm_inlier, index_10, perm_sparse = model(graph.x1, graph.edge_index1, graph.rel1, None, 
                                                        graph.x2, graph.edge_index2, graph.rel2, None, 
                                                        graph.map1, graph.map2)
        hist1_noneigh = hit_at_1_sparse_batch(batch_id=num_clu, pmat_pred=perm_sparse, test_y=graph.gt_y, 
                                              num_src=num_src, num_tgt=num_tgt, indices_s=graph.assoc1, 
                                              indices_t=graph.assoc2)
        hit10 = hit_at_10_sparse_batch(batch_id=num_clu, pred_10=index_10, test_y=graph.gt_y,
                                        indices_s=graph.assoc1, indices_t=graph.assoc2, 
                                        map_s=graph.map1, map_t=graph.map2)
        
        hits1 += hist1_noneigh
        hits10 += hit10
        hits1_dict = dict()
        hits1_dict['hits1_noneigh'] = hist1_noneigh.item()
        tfboardwriter.add_scalars('his1_per_clu', hits1_dict, num_clu)
        num_clu += 1

        # TODO 只快速查看一下没匹配上的个数
        matched_src = torch.isin(graph.map1, perm_inlier._indices()[0])
        matched_tgt = torch.isin(graph.map2, perm_inlier._indices()[1])
        
        non_allocs_src_num += (~matched_src).sum()
        non_allocs_tgt_num += (~matched_tgt).sum()        
    
    print(f'non_allocs_src_num: {non_allocs_src_num}, non_allocs_tgt_num: {non_allocs_tgt_num}')
    hits1 = hits1 / torch.tensor(graph.gt_y.size(1)).to(device)
    hits10 = hits10 / torch.tensor(graph.gt_y.size(1)).to(device)
    print(f'Hits@1: {hits1:.8f}', f'\tHits@10: {hits10:.8f}')
    eval_speed = time.time() - eval_since
    print(f"Finish evaluating in {eval_speed}s.")
    return model

        

if __name__ == '__main__':
    args = parse_args('Entity clustering then Alignment')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    datasets = eval(cfg.DATASET_FULL_NAME)

    for pair in ['de']:
    # for pair in ['zh_en']:
        print('*' * 20)
        print(pair)
        print('*' * 20)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model
        align_model = alignment_with_neighbor_nomask().to(device)
        align_optim = optim.Adam(align_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.LR_DECAY)
        
        if not Path(cfg.OUTPUT_PATH, pair).exists():
            Path(cfg.OUTPUT_PATH, pair).mkdir(parents=True)

        now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH, pair) / 'tensorboard' / 'training_{}'.format(now_time)))
        # subgraph cluster
        file_path_lp_reclustering = Path(cfg.OUTPUT_PATH, pair, 'lp_recluster_lists.pth')
        subgraph_parts = torch.load(file_path_lp_reclustering)
        subgraph_src, subgraph_tgt = subgraph_parts['cluster_src'], subgraph_parts['cluster_tgt']
        
        

        print("initial subgraphs id have loaded.")
        print('*' * 30)
        
        with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH, pair) / ('train_log_' + now_time + '.log'))) as _:
            print_easydict(cfg)
            # --- load data --- #
            print("Loading data ... ")
            if cfg.DATASET_FULL_NAME in ['mkdata', 'openeav2']:
                dataset = datasets(cfg.DATASET_PATH, pair, emb_model='Labse', outlier=False)[0]
            elif cfg.DATASET_FULL_NAME in ['DBP15K']:
                dataset = datasets(cfg.DATASET_PATH, pair)[0]

            # # ablation: directly k-means
            # subgraph_src, subgraph_tgt, (srcs_acc, tgts_acc) = clustering(dataset.x1, dataset.x2, 
            #                                         num_parts=cfg.NUM_CENTROIDS,
            #                                         gt_y=dataset.gt_y,
            #                                         clu_method='K-means', 
            #                                         # clu_method='Metis',
            #                                         # clu_method='groundtruth',
            #                                         edge_index1=dataset.edge_index1, 
            #                                         edge_index2=dataset.edge_index2,
            #                                         edge_rel1=dataset.rel1, 
            #                                         edge_rel2=dataset.rel2)
            
            # --- train --- #
            train_data = SuperClusterData(data=dataset, num_parts=len(subgraph_src), 
                                          cluster_list=(subgraph_src, subgraph_tgt),)
            train_loader = ClusterLoader(train_data, batch_size=1,)
            print("Finish loading train dataset.")

            # # 加neighbors, 确保num_src <= target
            # clus_src_neig, clus_tgt_neig, _, = add_neighbor_(num_parts=len(subgraph_src),
            #                                         clus_src=subgraph_src, clus_tgt=subgraph_tgt, 
            #                                         edge_index1=dataset.edge_index1,
            #                                         edge_index2=dataset.edge_index2,
            #                                         num_src_nodes=dataset.x1.size(0), num_tgt_nodes=dataset.x2.size(0))
                
            # train_data = SuperClusterData(data=dataset, num_parts=len(subgraph_src), 
            #                               cluster_list=(clus_src_neig, clus_tgt_neig),)
            # train_loader = ClusterLoader(train_data, batch_size=1,)
            # print("Finish loading train dataset.")
            
            
            align_model = train_eval(model=align_model, optimizer=align_optim, train_loader=train_loader, eval_loader=train_loader,
                              tfboardwriter=tfboardwriter, num_src=dataset.x1.size(0), num_tgt=dataset.x2.size(0), start_epoch=0, num_epochs=cfg.TRAIN.NUM_EPOCHS)
                
    print("Done.")
        