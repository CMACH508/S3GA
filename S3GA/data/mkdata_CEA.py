from typing import Callable, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index, subgraph

import re
import pickle
import codecs
import os.path as osp
from random import shuffle
from typing import Callable, Dict, List, Optional, Tuple

import scipy.sparse as sp
import numpy as np

def func(triples):
    """ 
    rel_weight = head 的种类 / rel在triples 中出现的次数 ()
    """
    head = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:  # relation
            cnt[tri[1]] = 1
            head[tri[1]] = {tri[0]}
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f

def ifunc(triples):
    """ 
    rel_weight = tail 的种类 / rel在triples 中出现的次数 ()
    """
    tail = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = {tri[2]}
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col))
        values = mx.data
        shape = mx.shape
        return torch.LongTensor(coords), torch.FloatTensor(values), shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

class mkdata(InMemoryDataset):
    pairs = ['fr', 'de']
    def __init__(self, root: str, pair: str, emb_model='dual-large',
                 outlier=True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.pair = pair
        self.emb_model = emb_model
        self.outlier = outlier
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')
    
    @property
    def processed_file_names(self) -> str:
        return f'{self.pair}_{self.emb_model}_outlier{self.outlier}.pt'

    def add_cnt_for(self, mp, val, begin=None):
        # val = re.sub('[_-]', ' ', val.split('/')[-1])
        if begin is None:
            if val not in mp:
                mp[val] = len(mp)
            return mp, mp[val]
        else:
            if val not in mp:
                mp[val] = begin
                begin += 1
            return mp, mp[val], begin
        
    def process_one_graph(self, rel_pos: str):
        triples, rel_idx, ent_idx = [], {}, {}
        with codecs.open(osp.join(self.raw_dir, 
                                  '{}_{}'.format(self.pair, rel_pos)), "r", 'utf-8') as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                ent_idx, s = self.add_cnt_for(ent_idx, now[0])
                rel_idx, p = self.add_cnt_for(rel_idx, now[1])
                ent_idx, o = self.add_cnt_for(ent_idx, now[2])
                triples.append([s, p, o])
        
        
        return rel_idx, ent_idx, triples
    
    def process_y(self, ent_links_path, ent1, ent2, filter_link=True):
        """ filter_link: 去掉不在ent1 和 ent2 中的link
        """
        link = []
        link1 = set()
        link2 = set()
        ent1_clean, ent2_clean = {}, {}
        with codecs.open(osp.join(self.raw_dir, '{}_{}'.format(self.pair, ent_links_path)), "r", 'utf-8') as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                if (now[0] in ent1 and now[1] in ent2) or (not filter_link):
                    ent1, src = self.add_cnt_for(ent1, now[0])
                    ent2, trg = self.add_cnt_for(ent2, now[1])
                    if src in link1 or trg in link2:
                        continue
                    link1.add(src)
                    link2.add(trg)
                    link.append((src, trg))


        with open(osp.join(self.raw_dir, f'{self.pair}_ent_1'), 'wb') as f:
            pickle.dump(ent1, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(osp.join(self.raw_dir, f'{self.pair}_ent_2'), 'wb') as f:
            pickle.dump(ent2, f, protocol=pickle.HIGHEST_PROTOCOL)
        return torch.tensor(link).T
    
    def process_graph(self, ent1, rel1, triple1, ent2, rel2, triple2, emb_model='dual-large'):
        if emb_model == 'dual-large':
            with open(osp.join(self.raw_dir, 'tmp', 'embeddings_large_{}.pkl'.format(self.pair)), 'rb') as f:
                embeddings = pickle.load(f)
                x1, x2 = embeddings
            with open(osp.join(self.raw_dir, 'tmp', 'ea_large_{}.pkl'.format(self.pair)), 'rb') as f:
                ea = pickle.load(f)
                # 一些小test 
                assert ea['ent1'] == ent1, "the tmp file is incorrect..."
                assert ea['ent2'] == ent2, "the tmp file is incorrect..."
        elif emb_model == 'Glove':
            embs = {}
            with open(osp.join(self.raw_dir, 'sub.glove.300d'), 'r') as f:
                for _, line in enumerate(f):
                    info = line.strip().split(' ')
                    if len(info) > 300:
                        embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
          
            with open(osp.join(self.raw_dir, f'{self.pair}_ent_1'), 'rb') as f:
                ent1_clean = pickle.load(f)
                ent1_vec = torch.zeros(len(ent1_clean), 300)

                for name, idx in ent1_clean.items():
                    k = 0
                    for word in re.sub('[_-]', ' ', name.split('/')[-1]).split():
                        word = word.lower()
                        if word in embs:
                            ent1_vec[idx] += embs[word]
                            k += 1
                    if k:
                        ent1_vec[idx] /= k
                    else:
                        ent1_vec[idx] = torch.rand(300) - 0.5
                    
                    ent1_vec[idx] = ent1_vec[idx]/ torch.linalg.norm(ent1_vec[idx])
            
            with open(osp.join(self.raw_dir, f'{self.pair}_ent_2'), 'rb') as f:
                ent2_clean = pickle.load(f)
                ent2_vec = torch.zeros(len(ent2_clean), 300)

                for name, idx in ent2_clean.items():
                    k = 0
                    for word in re.sub('[_-]', ' ', name.split('/')[-1]).split():
                        word = word.lower()
                        if word in embs:
                            ent2_vec[idx] += embs[word]
                            k += 1
                    if k:
                        ent2_vec[idx] /= k
                    else:
                        ent2_vec[idx] = torch.rand(300) - 0.5
                    
                    ent2_vec[idx] = ent2_vec[idx]/ torch.linalg.norm(ent2_vec[idx])

            x1 = ent1_vec
            x2 = ent2_vec                            

        elif emb_model == 'Labse':
            emb1_path = osp.join(self.raw_dir, f'{self.pair}_ent_labse_emb_1')
            emb2_path = osp.join(self.raw_dir, f'{self.pair}_ent_labse_emb_2')
            if osp.exists(emb1_path) and osp.exists(emb2_path):
                with open(emb1_path, 'rb') as f:
                    x1 = pickle.load(f)
                with open(emb2_path, 'rb') as f:
                    x2 = pickle.load(f)
            else:
                from transformers import BertModel, BertTokenizerFast
                tokenizer = BertTokenizerFast.from_pretrained("/dssg/home/acct-eetsk/eetsk/guowenqi/LaBSE")
                model = BertModel.from_pretrained("/dssg/home/acct-eetsk/eetsk/guowenqi/LaBSE")
                model = model.to('cuda')
                model = model.eval()
                
                with open(osp.join(self.raw_dir, f'{self.pair}_ent_1'), 'rb') as f:
                    ent1_clean = pickle.load(f)
                    ent1_vec = torch.zeros(len(ent1_clean), 768).to('cuda')
                    
                    for name, idx in ent1_clean.items():
                        inputs = tokenizer(name.split('/')[-1], return_tensors="pt", padding=True).to('cuda')
                        with torch.no_grad():
                            outputs = model(**inputs)
                            ent1_vec[idx] += outputs.pooler_output.squeeze(0)
        
                    print("ent1_vec is Done! ")
                with open(osp.join(self.raw_dir, f'{self.pair}_ent_2'), 'rb') as f:
                    ent2_clean = pickle.load(f)
                    ent2_vec = torch.zeros(len(ent2_clean), 768).to('cuda')
                    
                    for name, idx in ent2_clean.items():
                        inputs = tokenizer(name.split('/')[-1], return_tensors="pt", padding=True).to('cuda')
                        with torch.no_grad():
                            outputs = model(**inputs)
                            ent2_vec[idx] += outputs.pooler_output.squeeze(0)
                    
                    print("ent2_vec is Done! ")
                            
                x1 = ent1_vec.to('cpu')
                x2 = ent2_vec.to('cpu')
                with open(osp.join(self.raw_dir, f'{self.pair}_ent_labse_emb_1'), 'wb') as f:
                    pickle.dump(x1, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(osp.join(self.raw_dir, f'{self.pair}_ent_labse_emb_2'), 'wb') as f:
                    pickle.dump(x2, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            
        def get_weighted_rel(triple, Ns):
            # rel_weight
            r2f = func(triple)
            r2if = ifunc(triple)
            M ={}
            for tri in triple:
                if tri[0] == tri[2]:
                    continue
                if (tri[0], tri[2]) not in M:
                    M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
                else:
                    M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
                if (tri[2], tri[0]) not in M:
                    M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
                else:
                    M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
            
            row, col, data = [], [], []
            for key in M:
                row.append(key[1])
                col.append(key[0])
                data.append(M[key])
            
            return sp.coo_matrix((data, (row, col)), shape=(Ns, Ns))
            # edge_index = torch.stack((torch.tensor(row), torch.tensor(col)), dim=0)
            # rel = torch.tensor(data) 
            # edge_index, rel = sort_edge_index(edge_index=edge_index, edge_attr=rel)
            # return edge_index, rel
        
        adj1 = get_weighted_rel(triple1, Ns=x1.shape[0])
        adj2 = get_weighted_rel(triple2, Ns=x2.shape[0])
        edge_index1, rel1, _ = self.preprocess_adj(adj1)
        edge_index2, rel2, _ = self.preprocess_adj(adj2)

        return x1, edge_index1, rel1, x2, edge_index2, rel2

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        
        def normalize_adj(adj):
            """Symmetrically normalize adjacency matrix."""
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return sparse_to_tuple(adj_normalized)
    
    def process(self):
        rel1, ent1, triple1 = self.process_one_graph('triples_1')
        rel2, ent2, triple2 = self.process_one_graph('triples_2')
        gt_y = self.process_y('ent_links', ent1, ent2) # list
    
        x1, edge_index1, rel1, x2, edge_index2, rel2 = self.process_graph(ent1, rel1, triple1, ent2, rel2, triple2, self.emb_model)
        
        if self.outlier:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, x2=x2, 
                        edge_index2=edge_index2, rel2=rel2, gt_y=gt_y)
        else:
            # shuffle 
            col_y = torch.randperm(gt_y.size(1))
            x1 = x1[gt_y[0]] 
            x2 = x2[gt_y[1]]
            x1 = x1[col_y]
            edge_index1, rel1 = subgraph(subset=gt_y[0][col_y], edge_index=edge_index1, 
                                         edge_attr=rel1, relabel_nodes=True)
            edge_index2, rel2 = subgraph(subset=gt_y[1], edge_index=edge_index2, 
                                         edge_attr=rel2, relabel_nodes=True)
            gt_y = torch.stack([torch.arange(gt_y.size(1)), col_y], dim=0)
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1,
                        x2=x2, edge_index2=edge_index2, rel2=rel2, gt_y=gt_y)

        torch.save(self.collate([data]), self.processed_paths[0])



if __name__ == '__main__':
    path = osp.join('../data/', 'mkdata')
    # for pair in mkdata.pairs:
    #     data = mkdata(path, pair,  outlier=True)[0]
    #     print(data.x1.shape, data.x2.shape)
    #     print(data.gt_y.shape)
    for pair in mkdata.pairs:
        data = mkdata(path, pair, emb_model='Labse', outlier=False)[0]
        print(data.x1.shape, data.x2.shape)
        exit()
        print(data.edge_index1.shape, data.edge_index2.shape)
        import networkx as nx
        G1 = nx.Graph()
        G2 = nx.Graph()
        edge1 = data.edge_index1.T.tolist()
        edge2 = data.edge_index2.T.tolist()
        edge1 = tuple(map(tuple, edge1))
        edge2 = tuple(map(tuple, edge2))
        
        G1.add_edges_from(edge1)
        G2.add_edges_from(edge2)
        subgraph1 = nx.connected_components(G1)
        subgraph2 = nx.connected_components(G2)
        component1 = torch.tensor(list(next(subgraph1)))
        component2 = torch.tensor(list(next(subgraph2)))
        print(torch.isin(data.gt_y[0], component1).sum(), data.gt_y[0].shape)
        print(torch.isin(data.gt_y[1], component2).sum(), data.gt_y[1].shape)
        # print(pair,[len(c) for c in subgraph1], [len(c) for c in subgraph2])

        # print(data.x1.norm(dim=-1), data.x2.norm(dim=-1))
        # print(data.x1.shape, data.x2.shape)
        # print(data.edge_index1.shape, data.edge_index2.shape)
        # print(data.gt_y.shape)
        # from utils.data_clustering import clustering
        # clus_src, clus_tgt = clustering(data.x1, data.x2, 
        #                                     num_parts=30,
        #                                     gt_y=data.gt_y,
        #                                     clu_method='K-means', 
        #                                     edge_index1=data.edge_index1, 
        #                                     edge_index2=data.edge_index2)