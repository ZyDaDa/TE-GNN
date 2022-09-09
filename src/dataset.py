from cmath import exp, tau
from time import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np

def load_data(args):
    dataset_folder = os.path.abspath(os.path.join('dataset'))

    train_set = SeqDataset(dataset_folder,'train')
    test_set = SeqDataset(dataset_folder,'test')

    train_loader = DataLoader(train_set,args.batch_size,  num_workers=0,
                              shuffle=True,collate_fn=collate_fn,drop_last=True)
    test_loader = DataLoader(test_set,args.batch_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)

    id_maps = pickle.load(open(os.path.join( dataset_folder,'idmap.pkl'), 'rb'))
    item_num = max(id_maps[0].values())+1
    
    return train_loader, test_loader, item_num

class SeqDataset(Dataset):
    def __init__(self, datafolder, file='train',max_len=50) -> None:
        super().__init__()
        self.max_len = max_len
        data_file = os.path.join(datafolder, file+'.pkl')

        self.data = pickle.load(open(data_file,'rb')) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        # get raw data
        session = self.data[index]['items'][-self.max_len:]
        timestamp = self.data[index]['timestamp'][-self.max_len:]
        target = self.data[index]['target']
        # construct graph
        item_node = np.unique(session)
        item_map = dict([(i,idx) for idx,i in enumerate(item_node)]) # pyg data format

        item2idx = [item_map[i] for i in session] # convert itemid to index in item_node

        # compute edge weight
        D_init, D_final = 0.98, 0.01
        m = max(timestamp[-1] - timestamp[0],1) # eq.4
        alpha = np.log(D_init / D_final) / m # eq.3 
        l = -np.log(D_init) / alpha # eq.2

        item_edge = [[],[]] 
        item_edge_weight = []
        for h,t, ti,tj in zip(item2idx[:-1],item2idx[1:], timestamp[:-1],timestamp[1:]):
            tau = np.exp((-alpha*(abs(ti-tj)+l))) 
            item_edge[0].append(h)
            item_edge[1].append(t)
            item_edge_weight.append(tau)


        q = [ timestamp[-1]-t for t in timestamp]
        
        gamma = np.exp(-1*alpha*(np.array(q) + l)) # eq.8
        M = 6 # bins number
        mu = (np.exp(-alpha*l) - np.min(gamma)) / M # eq.9

        if len(session) == 1: bin = [M] # when sesion length == 1, bin == nan
        else:
            bin = []
            for g in gamma:
                bin.append(min(g//(mu+1e-8)  + 1, M))

        return item_node, item2idx, item_edge, item_edge_weight, bin, target

def collate_fn(batch_data):
    max_len = max([ len(d[-2]) for d in batch_data]) # max session length in batch

    batch_item_nodes = [] # 1d 
    batch_item2idx = [] # 1d
    batch_item_edge = [] # n*2
    batch_item_edge_weight = []
    batch_bins = []

    batch_session_len = [] # 1d , split above tensor by this term

    batch_target = []

    now_item_idx = 0
    for d in batch_data:
        item_node, item2idx, item_edge, item_edge_weight, bin, target = d

        batch_item_nodes.append(torch.LongTensor(item_node))

        batch_item2idx.append(torch.LongTensor(item2idx)+now_item_idx)

        batch_item_edge.append(torch.LongTensor(item_edge)+now_item_idx)

        batch_item_edge_weight.append(torch.FloatTensor(item_edge_weight))
        batch_bins.append(torch.LongTensor([0]*(max_len-len(bin))+bin)) # padding 0 

        batch_session_len.append(len(item2idx))
        batch_target.append(target)

        now_item_idx += len(item_node)

    batch_item_nodes = torch.concat(batch_item_nodes)
    batch_item2idx = torch.concat(batch_item2idx)
    batch_item_edge = torch.concat(batch_item_edge,dim=1)
    batch_item_edge_weight = torch.concat(batch_item_edge_weight, dim=-1)
    batch_bins = torch.stack(batch_bins,0)
    batch_target = torch.LongTensor(batch_target)

    batch = {}
    batch['items'] = batch_item_nodes
    batch['item2idx'] = batch_item2idx
    batch['edge'] = batch_item_edge
    batch['edge_weight'] = batch_item_edge_weight
    batch['bins'] = batch_bins
    batch['sess_len'] = batch_session_len
    batch['target'] = batch_target

    return batch