import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv
from model.GIN.readout import SumPooling, AvgPooling, MaxPooling
from model.Norm.norm import Norm
from model.Norm.norm import ContNorm
from model.GCN.embedding import *


class GCN(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, graph_pooling_type, norm_type='gn'):
        super(GCN, self).__init__()
        #TODO number of classes
        self.num_layers = num_layers

        self.gcnlayers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.gcnlayers.append(GraphConv(input_dim, hidden_dim))
            else:
                self.gcnlayers.append(
                    GraphConv(hidden_dim, hidden_dim)
                )
            if norm_type == 'bn' or norm_type == 'gn': 
                self.norms.append(Norm(norm_type, hidden_dim))
            
            if norm_type == 'cont':
                self.norms.append(ContNorm(act=None, norm= "batch", ch= hidden_dim, emb_dim = hidden_dim, spectral=False, no_act=True, twod=False))

        
        if norm_type == 'cont':
            self.TimeNet_1 = embedding.Time_Embedding(num_classes, hidden_dim, label=False, normalize=False, normalize_spectral=False)
            self.LabelNet_1 = embedding.Time_Embedding(num_classes, hidden_dim, label=True, normalize=False, normalize_spectral=False)
        
        self.linears_prediction = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(final_dropout)
        
                
        
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self,  g, h, t=None, y=None, std=None):
        hidden_rep = [h]
        split_list = g.batch_num_nodes
        
        if t!=None:
            t_emb = self.TimeNet_1(t)
            y_emb = self.LabelNet_1(y)
        
        for i in range(self.num_layers - 1):
            x = h
            h = self.gcnlayers[i](g, h)
            if t!=None:
                h = self.norms[i](h, t_emb+y_emb)
            else:
                h = self.norms[i](g, h)
            if i != 0:
                h = F.relu(h) + x
            else:
                h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        pooled_h = self.pool(g, hidden_rep[-1])
        score_over_layer += self.drop(self.linears_prediction(pooled_h))

        return score_over_layer