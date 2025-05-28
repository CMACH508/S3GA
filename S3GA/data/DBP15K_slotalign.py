import enum
import os
import os.path as osp
import random
import json
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index
import numpy as np

class DBP15K(InMemoryDataset):
    pairs = ['fr_en', 'ja_en', 'zh_en']
    def __init__(self, root:str, pair: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert pair in ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']
        self.pair = pair
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self) -> List[str]:
        return ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']
    
    @property
    def processed_file_names(self) -> str:
        return f'{self.pair}.pt'
    
    def process(self):
        embs = {}
        with open(osp.join(self.raw_dir, 'sub.glove.300d'), 'r') as f:
            for _, line in enumerate(f):
                info = line.strip().split(' ')
                if len(info) > 300:
                    embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
                
        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')
        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')
        x1_path = osp.join(self.raw_dir, self.pair, 'ent_ids_1')
        x2_path = osp.join(self.raw_dir, self.pair, 'ent_ids_2')
        x_path = osp.join(self.raw_dir, self.pair, 'dbp_'+self.pair+'.json')
        gt_path = osp.join(self.raw_dir, self.pair, 'ref_ent_ids')
        
        
        BERTfeature = np.load('/home/user/bi-ea/data/DBP15K_SEU_TRI/raw/dbp/LaBSE_{}.npy'.format(self.pair[:2]))
        feature = torch.tensor(BERTfeature)
        feature = feature / (feature.norm(dim=1)[:, None]+1e-16)
        node_size = feature.shape[0]
        file = np.load('/home/user/bi-ea/data/DBP15K_SEU_TRI/raw/dbp/KG{}.npz'.format(self.pair[:2]))
        sparse_rel_matrix = file['mat']
        sparse_rel_matrix = torch.sparse_coo_tensor(sparse_rel_matrix[:,:2].T, np.ones_like(sparse_rel_matrix[:,2]), (node_size,node_size)).float()
        

        # scr_entities and tgt_entities 
        with open(x1_path) as f:
            src_entities = f.readlines()
            
        with open(x2_path) as f:
            tgt_entities = f.readlines()
        
        src_entities = np.array([line.strip().split("\t")[0] for line in src_entities]).astype(np.int64)    
        tgt_entities = np.array([line.strip().split("\t")[0] for line in tgt_entities]).astype(np.int64)
        np.random.shuffle(src_entities)
        np.random.shuffle(tgt_entities)
        
        assoc1 = np.full((src_entities.max()+1, ), -1, dtype=np.int64)
        print(assoc1.shape, src_entities.shape, src_entities)
        assoc1[src_entities] = np.arange(src_entities.size)

        assoc2 = np.full((tgt_entities.max()+1, ), -1, dtype=np.int64)
        assoc2[tgt_entities] = np.arange(tgt_entities.size)

        with open(gt_path) as f:
            aligned = f.readlines()
        
        aligned = np.array([line.replace("\n", "").split("\t") for line in aligned]).astype(np.int64)
        aligned_row = torch.from_numpy(assoc1[aligned[:, 0]])
        aligned_col = torch.from_numpy(assoc2[aligned[:, 1]])
        gt_y = torch.stack([aligned_row, aligned_col], dim=0)
        
        # src and tgt features
        feature_src = feature[src_entities]
        feature_tgt = feature[tgt_entities]
        sparse_rel_matrix = sparse_rel_matrix.to_dense()
        adj_src = (sparse_rel_matrix[src_entities][:, src_entities]).to_sparse()
        adj_tgt = (sparse_rel_matrix[tgt_entities][:, tgt_entities]).to_sparse()
        print(adj_src, adj_src.shape)
        
        data = Data(x1=feature_src.float(), edge_index1=adj_src.coalesce().indices(), rel1=adj_src.coalesce().values().float(), 
                    x2=feature_tgt.float(), edge_index2=adj_tgt.coalesce().indices(), rel2=adj_tgt.coalesce().values().float(), gt_y=gt_y)

        torch.save(self.collate([data]), self.processed_paths[0]) 
    
if __name__ == '__main__':
    path = osp.join('..', 'data', 'DBP15K_SEU_TRI')
    for pair in DBP15K.pairs:
        data = DBP15K(path, pair)[0]
        print(data.x1.dtype, data.rel1.dtype, data.edge_index1.dtype, data.gt_y.dtype)
        print(data.x1.shape)
        
