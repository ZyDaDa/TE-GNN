from configparser import BasicInterpolation
import torch
from torch import nn
import numpy as np
import math
from GNN import T_GCN

class TE_GNN(nn.Module):
    def __init__(self,args, num_item) -> None:
        super().__init__()
        self.dim = args.dim

        self.item_embedding = nn.Embedding(num_item, args.dim)

        self.bin_embedding = nn.Embedding(10,args.dim, padding_idx=0)

        self.t_gcn = T_GCN(args.dim, args.layer_num)

        self.layer_num = args.layer_num

        self.dropout = nn.Dropout(args.dropout_rate)
        self.gru = nn.GRU(args.dim, args.dim, 
                            num_layers=1, 
                            batch_first=True,
                            bidirectional=True)
        self.w1 = nn.Linear(args.dim*2, 1,bias=False) # author's code setting

        self.w2 = nn.Linear(args.dim*2, args.dim, bias=False)
        self.v3 = nn.Linear(args.dim,1, bias=False)

        self.short_gru = nn.GRU(args.dim, args.dim, 
                            num_layers=1, 
                            batch_first=True)

        self.w3 = nn.Linear(args.dim, 1) # author's code setting

        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def forward(self, batch):

        h = self.item_embedding(batch['items'])

        h_final = self.t_gcn(h, batch['edge'], batch['edge_weight'])[batch['item2idx']]
        h_final = torch.dropout(h_final, p=0.3,train=self.training) # author's code setting

        # asym-BiGRU
        e = self.dropout(self.bin_embedding(batch['bins']))

        e_dir, _ = self.gru(e)

        b_gate = torch.sigmoid(self.w1(e_dir))
        e_hat = b_gate*e_dir[:,:,:self.dim] + (1-b_gate)*e_dir[:,:,self.dim:]

        e_hat = torch.concat([e[-l:] for e,l in zip(e_hat, batch['sess_len'])],dim=0)

        # attention network
        beta = self.v3(torch.tanh(self.w2(torch.concat([h_final,e_hat],dim=-1))))
        c = beta*h_final

        # recommendation
        per_sess_emb = torch.split(c,batch['sess_len']) # split tensor by session length
        z_long = torch.stack([e.sum(0) for e in per_sess_emb],dim=0)
        z_long = torch.dropout(z_long,p=0.3,train=self.training) # author's code setting

        c_pad = torch.zeros((len(batch['sess_len']),max(batch['sess_len']),c.size(-1))).to(c.device)
        for idx, c_emb in  enumerate(per_sess_emb):
            c_pad[idx,-len(c_emb):] = c_emb
        _, c_hat = self.short_gru(c_pad)

        z_short = c_hat.squeeze()

        f = torch.sigmoid(self.w3(z_long + z_short))
        z_final = f*z_long + (1-f)*z_short

        return torch.matmul(z_final, self.item_embedding.weight.T)
