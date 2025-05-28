import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import subgraph
from GSSL.clustering import Encoder
from utils.config import cfg
from torch_geometric.nn import LabelPropagation

class NodeClassification(nn.Module):
    def __init__(self, gnn_model="relgcn") -> None:
        super(NodeClassification, self).__init__()
        self.n_layers = cfg.MODEL.NUM_LAYER
        self.encoder = Encoder(in_feats=cfg.MODEL.IN_CHANNEL, n_hidden=cfg.MODEL.HIDDEN_CHANNEL, n_layers=cfg.MODEL.NUM_LAYER, gnn_model=gnn_model)
        self.fc = nn.Linear(in_features=cfg.MODEL.HIDDEN_CHANNEL*(self.n_layers+1), out_features=cfg.NUM_CENTROIDS)
    
    def forward(self, x_s, edge_index_s):
        xs = self.encoder(x_s, edge_index_s, norm=True)
        xs = self.fc(torch.cat(xs, dim=-1))
        return xs


class Metis:
    '''
    Metis clustering algorithm implemented with PyTorch
    Parameters:
      n_clusters: int,
        Number of clusters
    '''
    def __init__(self, n_clusters) -> None:
        self.n_clusters = n_clusters
        # self.gnn = SGFormer(in_channels=cfg.MODEL.IN_CHANNEL, 
        #                     hidden_channels=cfg.MODEL.HIDDEN_CHANNEL, 
        #                     out_channels=cfg.NUM_CENTROIDS)
        self.gnn = NodeClassification()
    
    def fit(self, x, edge_index, edge_weight=None, num_epochs=100, 
            weight_decay=0., lr=0.001, batch_training=True, nx=True):
        N = x.size(0)      
        # import networkx as nx
        # import nxmetis
        # g = nx.Graph()
        # edges = edge_index.cpu().numpy()
        # edges = zip(*edges)
        # g.add_edges_from(edges)
        # nx.set_edge_attributes(g, 1, 'weight')
        # _, partition = nxmetis.partition(g, self.n_clusters)
        # y = torch.full((N,), fill_value=-1, dtype=torch.int64)
        # for i in range(self.n_clusters):
        #     y[partition[i]] = i
        from torch_geometric.utils import to_undirected
        tmp_index = to_undirected(edge_index)
        E = tmp_index.size(1)
        adj = SparseTensor(
            row=tmp_index[0], col=tmp_index[1], 
            value=torch.ones(E, device=edge_index.device).to(torch.long),
            sparse_sizes=(N, N))
        rowptr, col, value = adj.csr()
        rowptr, col = rowptr.cpu(), col.cpu()
        y = torch.ops.torch_sparse.partition(rowptr, col, value,
                                            self.n_clusters, False)

        
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(
                self.gnn.parameters(), weight_decay=weight_decay, lr=lr)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        for epoch in range(num_epochs):
            if not batch_training:
                self.gnn.to(device)
                self.gnn.train()
                optimizer.zero_grad()
                x = x.to(device)
                edge_index = edge_index.to(device)
                # edge_weight = edge_weight.to(device)
                y = y.to(device)
                out = self.gnn(x, edge_index)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            else: 
                self.gnn.to(device)
                self.gnn.train()
                
                num_batch = N // cfg.BATCH_SIZE + (N%cfg.BATCH_SIZE>0)
                idx = torch.randperm(N)
                for i in range(num_batch):
                    idx_i = idx[i*cfg.BATCH_SIZE:(i+1)*cfg.BATCH_SIZE]
                    x_i = x[idx_i].to(device)
                    edge_index_i, edge_weight_i = subgraph(idx_i, edge_index, 
                                                           edge_attr=edge_weight, 
                                                           num_nodes=N, relabel_nodes=True)
                    edge_index_i = edge_index_i.to(device)
                    # edge_weight_i = edge_weight_i.to(device)
                    y_i = y[idx_i].to(device)
                    optimizer.zero_grad()
                    out_i = self.gnn(x_i, edge_index_i)
                    out_i = F.log_softmax(out_i, dim=1)
                    loss = criterion(out_i, y_i)
                    
                    loss.backward()
                    optimizer.step()
        
        
            if epoch % 5 == 0:
                print('In epoch {}, loss: {:.3f}'.format(
                    epoch, loss))  
        return y.to("cpu")
    
    def fit_predict(self, x, edge_index, train_mask, label, num_epochs=50, batch_training=True):
        N = x.size(0)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # x = x.to(device)
        # self.gnn.to(device)
        # edge_index = edge_index.to(device)
        # self.gnn.eval()
        # out = self.gnn(x, edge_index)
        # out = out.argmax(dim=-1)

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(
                self.gnn.parameters(), weight_decay=0.0, lr=0.001)
        for epoch in range(num_epochs):
            self.gnn.to(device)
            self.gnn.train()
            if not batch_training:
                optimizer.zero_grad()
                x = x.to(device)
                edge_index = edge_index.to(device)
                # edge_weight = edge_weight.to(device)
                label = label.to(device)
                out = self.gnn(x, edge_index)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_mask], label)
                loss.backward()
                optimizer.step()
                out = out.argmax(dim=1)

            else:
                mask = torch.zeros(N, dtype=torch.bool)
                mask[train_mask] = True
                y = torch.full((N,), -1)
                y[train_mask] = label

                num_batch = N // cfg.BATCH_SIZE + (N%cfg.BATCH_SIZE>0)
                idx = torch.randperm(N)
                out = torch.full_like(idx, -1)

                for i in range(num_batch):  
                    idx_i = idx[i*cfg.BATCH_SIZE:(i+1)*cfg.BATCH_SIZE]
                    x_i = x[idx_i].to(device)
                    edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=N, relabel_nodes=True)
                    edge_index_i = edge_index_i.to(device)
                    # edge_weight_i = edge_weight_i.to(device)
                    train_mask_i = mask[idx_i]
                    y_i = y[idx_i].to(device)

                    optimizer.zero_grad()
                    out_i = self.gnn(x_i, edge_index_i)
                    out_i = F.log_softmax(out_i, dim=1)
                    loss = criterion(out_i[train_mask_i], y_i[train_mask_i])
                    loss.backward()
                    optimizer.step()

                    
            if epoch % 5 == 0:
                print('In epoch {}, loss: {:.3f}'.format(
                    epoch, loss)) 
            if epoch == num_epochs-1:
                
                self.gnn.eval()
                x = x.to(device)
                edge_index = edge_index.to(device)
                out = self.gnn(x, edge_index)
                out = out.argmax(dim=1)
            
            
        return  out.to("cpu")

class Metis_LP:
    def __init__(self, n_clusters) -> None:
        self.n_clusters = n_clusters
        self.gnn = LabelPropagation(num_layers=3, alpha=0.9)
    
    def fit(self, x, edge_index):
        N = x.size(0) 
        from torch_geometric.utils import to_undirected, remove_self_loops
        tmp_index = to_undirected(edge_index)
        tmp_index, _ = remove_self_loops(tmp_index)
        E = tmp_index.size(1)
        adj = SparseTensor(
            row=tmp_index[0], col=tmp_index[1], 
            value=torch.ones(E, device=edge_index.device).to(torch.long),
            sparse_sizes=(N, N))
        rowptr, col, value = adj.csr()
        rowptr, col = rowptr.cpu(), col.cpu()
        y = torch.ops.torch_sparse.partition(rowptr, col, value,
                                            self.n_clusters, False)
        return y.to("cpu")
    
    def fit_predict(self, x, edge_index, train_mask, label):
        N = x.size(0)
        # y = torch.full((N,), -1)
        y = torch.randint(0, self.n_clusters, (N,))
        y[train_mask] = label
        out = self.gnn(y, edge_index, mask=train_mask)
        out = out.argmax(dim=-1)
        return out.to("cpu")