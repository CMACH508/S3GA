import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch


def to_sparse(x, mask):
    return x[mask]

class RelConv_(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv_, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Lin(in_channels, out_channels, bias=False)
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index,  edge_weight=None, x_cen=None):
        """"""
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x), edge_weight=edge_weight)
        # self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x), edge_weight=edge_weight)
        return self.root(x_cen) + out1 + out2 if x_cen is not None else self.root(x) + out1 + out2
        # return self.root(x_cen) + out1 if  x_cen is not None else self.root(x) + out1
    
    def message(self, x_j, edge_weight=None):
        return  x_j * edge_weight if edge_weight is not None else x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

