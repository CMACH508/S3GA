from typing import Optional
import copy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import subgraph, to_dense_batch, degree
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import SplineConv, GINEConv
from GSSL.rel import RelConv_

from utils.sinkhorn import Sinkhorn_sparse_rpcl, Sinkhorn_sparse
# from utils.hungarian import hungarian_sparse

from utils.config import cfg

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, gnn_model="relconv"):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.mlp1 = nn.Sequential(
            nn.Linear(in_feats, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        # self.bns = nn.ModuleList()
        # self.bns.append(nn.BatchNorm1d(in_feats))
        # self.fcs = nn.ModuleList()
        # self.fcs.append(nn.Linear(in_feats, n_hidden))
        for i in range(self.n_layers):
            if i == 0:
                if gnn_model == "relgcn":
                    gnn_layer = RelConv_(in_channels=in_feats, out_channels=n_hidden)
                    # gnn_layer = GINEConv(nn=self.mlp1, train_eps=True, edge_dim=1)
                    # gnn_layer = SplineConv(in_channels=in_feats, out_channels=n_hidden, dim = 1, kernel_size=5)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            else:
                if gnn_model == "relgcn":
                    gnn_layer = RelConv_(in_channels=n_hidden, out_channels=n_hidden)
                    # gnn_layer = GINEConv(nn=self.mlp2, train_eps=True, edge_dim=1)
                    # gnn_layer = SplineConv(in_channels=n_hidden, out_channels=n_hidden, dim = 1, kernel_size=5)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            # self.bns.append(nn.BatchNorm1d(n_hidden))

    def forward(self, x, edge_index, edge_attr=None, norm=False,):
        if norm:
            x = F.normalize(x, p=2, dim=-1)
        xs = [x]
        for i in range(self.n_layers):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            x = gnn_layer(x, edge_index)
            # x = F.elu(x)
            x = F.normalize(x, p=2, dim=-1)
            xs.append(x)
        
        return xs
    # def forward(self, x, edge_index, edge_attr=None, norm=False):
    #     if norm:
    #         # x = F.normalize(x, p=2, dim=-1)
    #         x = self.bns[0](x)
    #     x = self.fcs[0](x)
    #     xs = [x]
    #     for i in range(self.n_layers):
    #         gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
    #         x = gnn_layer(x, edge_index)
    #         x = self.bns[i+1](x)
    #         x = F.relu(x)
    #         x = F.dropout(x, p=0.3, training=self.training)
    #         xs.append(x)

    #     return xs
            


# class alignment_with_neighbor_nomask(nn.Module):
#     def __init__(self, gnn_model="relgcn") -> None:
#         super(alignment_with_neighbor_nomask, self).__init__()
#         self.n_layers = cfg.MODEL.NUM_LAYER
#         self.encoder = Encoder(in_feats=cfg.MODEL.IN_CHANNEL, n_hidden=cfg.MODEL.HIDDEN_CHANNEL, n_layers=cfg.MODEL.NUM_LAYER, gnn_model=gnn_model)
#         # self.encoder = Encoder(in_feats=768, n_hidden=256, n_layers=2, gnn_model=gnn_model)
#         self.sinkhorn1 = Sinkhorn_sparse(max_iter=10)
#         self.sinkhorn = Sinkhorn_sparse_rpcl(max_iter=10)
        
#     def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, 
#                       x_t, edge_index_t, edge_attr_t, batch_t,
#                       node_mapping_s, node_mapping_t):
        
#         # x_s = F.normalize(x_s, p=2, dim=-1)
#         # x_t = F.normalize(x_t, p=2, dim=-1)
#         # unary_affs = torch.einsum("nd,md->nm", x_s, x_t)
        
#         unary_affs = 0.0
#         xs = self.encoder(x_s, edge_index_s, edge_attr_s.unsqueeze(1), norm=True)
#         xt = self.encoder(x_t, edge_index_t, edge_attr_t.unsqueeze(1), norm=True)

#         for i in range(0, self.n_layers+1):
#             unary_affs += torch.einsum("nd, md->nm", xs[i], xt[i])
#         unary_affs /= (self.n_layers + 1)

#         # unary_affs = torch.einsum("nd, md->nm", xs[-1], xt[-1])
        
#         if x_s.shape[0] >= x_t.shape[0]:
#             unary_affs = F.normalize(unary_affs, p=2, dim=-1)
#         else:
#             unary_affs = F.normalize(unary_affs, p=2, dim=-2)
        
#         probability = unary_affs
#         _, binary_m, index_10 = self.sinkhorn(probability.detach())

#         pseudo_index_ori = binary_m._indices()
#         index_s = torch.isin(pseudo_index_ori[0], node_mapping_s)
#         index_t = torch.isin(pseudo_index_ori[1], node_mapping_t)
#         '''
#         xxxxxxxooo      xx: mask_
#         xxxxxxxooo      oo: mask      
#         xxxxxxxooo      xx+oo: mask_neig
#         ooooooonnn
#         ooooooonnn
#         '''
#         mask_neig = torch.logical_or(index_s, index_t)
#         pseudo_m = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_neig],
#                                                values=binary_m._values()[mask_neig],
#                                                 size=binary_m.size(), device=binary_m.device)
        
#         mask_ = torch.logical_and(index_s, index_t)
#         pseudo_m_ = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_],
#                                                values=binary_m._values()[mask_],
#                                                 size=binary_m.size(), device=binary_m.device)


#         # mask = torch.logical_xor(index_s, index_t)
#         # neig_m = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask],
#         #                                        values=binary_m._values()[mask],
#         #                                         size=binary_m.size(), device=binary_m.device)
#         # 考虑在evaluation的时候添加tgt的neighbor作为待匹配的点
#         '''
#         xxxxxxxppp      xxx+ppp: mask_src
#         xxxxxxxppp             
#         xxxxxxxppp      
#         ooooooonnn
#         ooooooonnn
#         '''
#         mask_src = index_s
#         neig_src = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_src],
#                                            values=binary_m._values()[mask_src],
#                                            size=binary_m.size(), device=binary_m.device)

#         # return [xs, xt], probability, pseudo_m if self.training else pseudo_m_, mask_neig, neig_m
#         return [xs, xt], unary_affs, binary_m if self.training else pseudo_m_, binary_m, neig_src
    
class alignment_with_neighbor_nomask(nn.Module):
    def __init__(self, gnn_model="relgcn") -> None:
        super(alignment_with_neighbor_nomask, self).__init__()
        self.n_layers = cfg.MODEL.NUM_LAYER
        self.encoder = Encoder(in_feats=cfg.MODEL.IN_CHANNEL, n_hidden=cfg.MODEL.HIDDEN_CHANNEL, n_layers=cfg.MODEL.NUM_LAYER, gnn_model=gnn_model)
        # self.encoder = Encoder(in_feats=768, n_hidden=256, n_layers=2, gnn_model=gnn_model)
        self.sinkhorn = Sinkhorn_sparse(max_iter=10)
        # self.sinkhorn = Sinkhorn_sparse_rpcl(max_iter=10)
        
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, 
                      x_t, edge_index_t, edge_attr_t, batch_t,
                      node_mapping_s, node_mapping_t):
        
        # x_s = F.normalize(x_s, p=2, dim=-1)
        # x_t = F.normalize(x_t, p=2, dim=-1)
        # unary_affs = torch.einsum("nd,md->nm", x_s, x_t)
        unary_affs = 0.0
        xs = self.encoder(x_s, edge_index_s, edge_attr_s.unsqueeze(1), norm=True)
        xt = self.encoder(x_t, edge_index_t, edge_attr_t.unsqueeze(1), norm=True)

        for i in range(0, self.n_layers+1):
            unary_affs += torch.einsum("nd, md->nm", xs[i], xt[i])
        unary_affs /= (self.n_layers + 1)

        # unary_affs = torch.einsum("nd, md->nm", xs[-1], xt[-1])
        
        if x_s.shape[0] >= x_t.shape[0]:
            unary_affs = F.normalize(unary_affs, p=2, dim=-1)
        else:
            unary_affs = F.normalize(unary_affs, p=2, dim=-2)
        
        probability = unary_affs
        _, binary_m, index_10 = self.sinkhorn(probability.detach())

        pseudo_index_ori = binary_m._indices()
        index_s = torch.isin(pseudo_index_ori[0], node_mapping_s)
        index_t = torch.isin(pseudo_index_ori[1], node_mapping_t)
        '''
        xxxxxxxooo      xx: mask_
        xxxxxxxooo      oo: mask      
        xxxxxxxooo      xx+oo: mask_neig
        ooooooonnn
        ooooooonnn
        '''
        # mask_neig = torch.logical_or(index_s, index_t)
        # pseudo_m = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_neig],
        #                                        values=binary_m._values()[mask_neig],
        #                                         size=binary_m.size(), device=binary_m.device)
        
        mask_ = torch.logical_and(index_s, index_t)
        pseudo_m_ = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_],
                                               values=binary_m._values()[mask_],
                                                size=binary_m.size(), device=binary_m.device)


        # mask = torch.logical_xor(index_s, index_t)
        # neig_m = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask],
        #                                        values=binary_m._values()[mask],
        #                                         size=binary_m.size(), device=binary_m.device)
        # 考虑在evaluation的时候添加tgt的neighbor作为待匹配的点
        '''
        xxxxxxxppp      xxx+ppp: mask_src
        xxxxxxxppp             
        xxxxxxxppp      
        ooooooonnn
        ooooooonnn
        '''
        mask_src = index_s
        neig_src = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_src],
                                           values=binary_m._values()[mask_src],
                                           size=binary_m.size(), device=binary_m.device)
        # print(index_10.shape, node_mapping_s.shape)

        # return [xs, xt], probability, pseudo_m if self.training else pseudo_m_, mask_neig, neig_m
        return [xs, xt], unary_affs, binary_m if self.training else pseudo_m_, index_10[node_mapping_s] if not self.training else None, neig_src
    

  
    
class ablation_alignment_with_neighbor_nomask(nn.Module):
    def __init__(self, gnn_model="relgcn") -> None:
        super(ablation_alignment_with_neighbor_nomask, self).__init__()
        self.n_layers = cfg.MODEL.NUM_LAYER
        # self.encoder = Encoder(in_feats=cfg.MODEL.IN_CHANNEL, n_hidden=cfg.MODEL.HIDDEN_CHANNEL, n_layers=cfg.MODEL.NUM_LAYER, gnn_model=gnn_model)
        self.encoder = Encoder(in_feats=768, n_hidden=256, n_layers=2, gnn_model=gnn_model)
        self.sinkhorn = Sinkhorn_sparse_rpcl(max_iter=10)
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, 
                      x_t, edge_index_t, edge_attr_t, batch_t,
                      node_mapping_s, node_mapping_t):
        
        # x_s = F.normalize(x_s, p=2, dim=-1)
        # x_t = F.normalize(x_t, p=2, dim=-1)
        # unary_affs = torch.einsum("nd,md->nm", x_s, x_t)
        unary_affs = 0.0
        xs = self.encoder(x_s, edge_index_s, edge_attr_s.unsqueeze(1), norm=True)
        xt = self.encoder(x_t, edge_index_t, edge_attr_t.unsqueeze(1), norm=True)

        for i in range(0, self.n_layers+1):
            unary_affs += torch.einsum("nd, md->nm", xs[i], xt[i])
        
        unary_affs /= (self.n_layers + 1)
        # unary_affs = torch.einsum("nd, md->nm", xs[-1], xt[-1])
        
        if x_s.shape[0] >= x_t.shape[0]:
            unary_affs = F.normalize(unary_affs, p=2, dim=-1)
        else:
            unary_affs = F.normalize(unary_affs, p=2, dim=-2)
        
        probability = unary_affs
        _, binary_m, index_10 = self.sinkhorn(probability.detach())

        pseudo_index_ori = binary_m._indices()
        index_s = torch.isin(pseudo_index_ori[0], node_mapping_s)
        index_t = torch.isin(pseudo_index_ori[1], node_mapping_t)
        '''
        xxxxxxxooo      xx: mask_
        xxxxxxxooo      oo: mask      
        xxxxxxxooo      xx+oo: mask_neig
        ooooooonnn
        ooooooonnn
        '''
        mask_neig = torch.logical_or(index_s, index_t)
        pseudo_m = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_neig],
                                               values=binary_m._values()[mask_neig],
                                                size=binary_m.size(), device=binary_m.device)
        
        mask_ = torch.logical_and(index_s, index_t)
        pseudo_m_ = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_],
                                               values=binary_m._values()[mask_],
                                                size=binary_m.size(), device=binary_m.device)


        # mask = torch.logical_xor(index_s, index_t)
        # neig_m = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask],
        #                                        values=binary_m._values()[mask],
        #                                         size=binary_m.size(), device=binary_m.device)
        # 考虑在evaluation的时候添加tgt的neighbor作为待匹配的点
        '''
        xxxxxxxppp      xxx+ppp: mask_src
        xxxxxxxppp             
        xxxxxxxppp      
        ooooooonnn
        ooooooonnn
        '''
        mask_src = index_s
        neig_src = torch.sparse_coo_tensor(indices=binary_m._indices()[:, mask_src],
                                           values=binary_m._values()[mask_src],
                                           size=binary_m.size(), device=binary_m.device)

        # return [xs, xt], probability, pseudo_m if self.training else pseudo_m_, mask_neig, neig_m
        return [xs, xt], unary_affs, binary_m if self.training else pseudo_m_, binary_m, neig_src
    
    @torch.no_grad()
    def embed(self, x_s, edge_index_s, edge_attr_s, batch_s,
                    x_t, edge_index_t, edge_attr_t, batch_t,
                    node_mapping_s, node_mapping_t):
        unary_affs = torch.einsum("nd, md->nm", x_s, x_t)
        if x_s.shape[0] >= x_t.shape[0]:
            unary_affs = F.normalize(unary_affs, p=2, dim=-1)
        else:
            unary_affs = F.normalize(unary_affs, p=2, dim=-2)
        probability = unary_affs
        _, binary_m, index_10 = self.sinkhorn(probability.detach())

        pseudo_index_ori = binary_m._indices()
        index_s = torch.isin(pseudo_index_ori[0], node_mapping_s)
        index_t = torch.isin(pseudo_index_ori[1], node_mapping_t)
        
        mask_ = torch.logical_and(index_s, index_t)
        pseudo_m_ = binary_m._indices()[:, mask_]
        candidate_s = torch.isin(node_mapping_s, pseudo_m_[0])
        candidate_t = torch.isin(node_mapping_t, pseudo_m_[1])
        
        c_center, non_emb_xs, non_emb_xt = [], [], []
        non_allocated_s = node_mapping_s[~candidate_s]
        non_allocated_t = node_mapping_t[~candidate_t]
        c_cent = torch.mean(torch.cat((x_s[node_mapping_s[candidate_s]],
                                       x_t[node_mapping_t[candidate_t]]), dim=0), dim=0)
        
        if not torch.isnan(c_cent).any():
            c_center.append(c_cent)
        non_emb_xs.append(x_s[non_allocated_s])
        non_emb_xt.append(x_t[non_allocated_t])
        
        return c_center, node_mapping_s[candidate_s], node_mapping_t[candidate_t], non_allocated_s, non_allocated_t, non_emb_xs, non_emb_xt, pseudo_m_

class alignment_for_partition(nn.Module):
    def __init__(self, gnn_model='relgcn') -> None:
        super(alignment_for_partition, self).__init__()
        self.n_layers = cfg.MODEL.NUM_LAYER
        self.encoder = Encoder(in_feats=768, n_hidden=256, n_layers=2, gnn_model=gnn_model)
        self.sinkhorn = Sinkhorn_sparse_rpcl(max_iter=10)
    
    @torch.no_grad()
    def embed(self, x_s, edge_index_s, edge_attr_s, batch_s, 
                      x_t, edge_index_t, edge_attr_t, batch_t, 
                      node_mapping_s, node_mapping_t):
        unary_affs = 0.0
        xs = self.encoder(x_s, edge_index_s, edge_attr_s.unsqueeze(1), norm=True)
        xt = self.encoder(x_t, edge_index_t, edge_attr_t.unsqueeze(1), norm=True)

        for i in range(0, self.n_layers+1):
            unary_affs += torch.einsum("nd, md->nm", xs[i], xt[i])
        
        unary_affs /= (self.n_layers + 1)
        if x_s.shape[0] >= x_t.shape[0]:
            unary_affs = F.normalize(unary_affs, p=2, dim=-1)
        else:
            unary_affs = F.normalize(unary_affs, p=2, dim=-2)
        probability = unary_affs
        _, binary_m, index_10 = self.sinkhorn(probability.detach())

        pseudo_index_ori = binary_m._indices()
        index_s = torch.isin(pseudo_index_ori[0], node_mapping_s)
        index_t = torch.isin(pseudo_index_ori[1], node_mapping_t)
        mask_ = torch.logical_and(index_s, index_t)
        pseudo_m_ = binary_m._indices()[:, mask_]
        candidate_s = torch.isin(node_mapping_s, pseudo_m_[0])
        candidate_t = torch.isin(node_mapping_t, pseudo_m_[1])
        
        c_center, non_emb_xs, non_emb_xt = [], [], []
        non_allocated_s = node_mapping_s[~candidate_s]
        non_allocated_t = node_mapping_t[~candidate_t]
        
        for i in range(self.n_layers + 1):
            c_cent = torch.mean(torch.cat((xs[i][node_mapping_s[candidate_s]], 
                                            xt[i][node_mapping_t[candidate_t]]), dim=0), dim=0)

            if not torch.isnan(c_cent).any():
                # c_center.append(c_cent.unsqueeze(0))
                c_center.append(c_cent)
            non_emb_xs.append(xs[i][non_allocated_s])
            non_emb_xt.append(xt[i][non_allocated_t])
        
        return c_center, node_mapping_s[candidate_s], node_mapping_t[candidate_t], non_allocated_s, non_allocated_t, non_emb_xs, non_emb_xt, pseudo_m_
 
class alignment_for_partition_two_cents(nn.Module):
    def __init__(self, gnn_model='relgcn') -> None:
        super(alignment_for_partition_two_cents, self).__init__()
        self.n_layers = cfg.MODEL.NUM_LAYER
        self.encoder = Encoder(in_feats=cfg.MODEL.IN_CHANNEL, n_hidden=cfg.MODEL.HIDDEN_CHANNEL,
                               n_layers=cfg.MODEL.NUM_LAYER, gnn_model=gnn_model)
        self.sinkhorn = Sinkhorn_sparse_rpcl(max_iter=10)
    
    @torch.no_grad()    
    def embed(self, x_s, edge_index_s, edge_attr_s, batch_s,
                    x_t, edge_index_t, edge_attr_t, batch_t,
                    node_mapping_s, node_mapping_t):
        unary_affs = 0.0
        xs = self.encoder(x_s, edge_index_s, edge_attr_s.unsqueeze(1), norm=True)
        xt = self.encoder(x_t, edge_index_t, edge_attr_t.unsqueeze(1), norm=True)
        
        for i in range(self.n_layers+1):
            unary_affs += torch.einsum("nd, md->nm", xs[i], xt[i])
        unary_affs /= (self.n_layers + 1)
        
        if x_s.shape[0] >= x_t.shape[0]:
            unary_affs = F.normalize(unary_affs, p=2, dim=-1)
        else:
            unary_affs = F.normalize(unary_affs, p=2, dim=-2)
        
        probability = unary_affs
        _, binary_m, index_10 = self.sinkhorn(probability.detach())
        
        pseudo_index_ori = binary_m._indices()
        index_s = torch.isin(pseudo_index_ori[0], node_mapping_s)
        index_t = torch.isin(pseudo_index_ori[1], node_mapping_t)
        mask_ = torch.logical_and(index_s, index_t)
        pseudo_m_ = binary_m._indices()[:, mask_]
        candidate_s = torch.isin(node_mapping_s, pseudo_m_[0])
        candidate_t = torch.isin(node_mapping_t, pseudo_m_[1])
        
        c_center_s, c_center_t, non_emb_xs, non_emb_xt = [], [], [], []
        non_allocated_s = node_mapping_s[~candidate_s]
        non_allocated_t = node_mapping_t[~candidate_t]
        
        for i in range(self.n_layers + 1):
            c_cent_s = torch.mean(xs[i][node_mapping_s[candidate_s]], dim=0)
            c_cent_t = torch.mean(xt[i][node_mapping_t[candidate_t]], dim=0)
            if not torch.isnan(c_cent_s).any:
                c_center_s.append(c_cent_s.unsqueeze(0))
                c_center_t.append(c_cent_t.unsqueeze(0))
            
            non_emb_xs.append(xs[i][non_allocated_s])
            non_emb_xt.append(xt[i][non_allocated_t])
            
        return [c_center_s, c_center_t], node_mapping_s[candidate_s], \
            node_mapping_t[candidate_t], non_allocated_s, \
                non_allocated_t, non_emb_xs, non_emb_xt, pseudo_m_
        
            
         

if __name__ == '__main__':
    model = Encoder(in_feats=768, n_hidden=256, n_layers=2, gnn_model="relgcn")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
                    