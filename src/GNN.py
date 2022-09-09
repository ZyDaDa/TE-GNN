import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

class T_GCN(MessagePassing):
    def __init__(self,dim:int,  layer_num:int,  aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.layer_num = layer_num
        # self.wg = nn.Linear(dim*2, dim,bias=False)
        self.wg = nn.Linear(dim*2, 1,bias=False) # author's code  setting 
        nn.init.normal_(self.wg.weight,std=0.1)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        h = x[:] # deepcopy
        for _ in range(self.layer_num):
            h_in = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            h_out = self.propagate(torch.stack([edge_index[1],edge_index[0]],dim=0), x=h, edge_weight=edge_weight)

            h = h_in + h_out + h # eq.5

        g = torch.sigmoid(self.wg(torch.concat([x,h],dim=-1)))

        h_final = g*h + (1-g)*x
        return h_final

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')
