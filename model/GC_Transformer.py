import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter
from model.Transformer_with_PE import Transformer_with_PE, PositionalEncoding
from torch_geometric.nn import ChebConv



class GC_Transformer(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, use_positional_encoding=True):
        super(GC_Transformer, self).__init__()
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_index = self.edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1) * city_num
        self.edge_index = self.edge_index.view(2, -1)
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.gcn_out = 14
        self.conv = ChebConv(self.in_dim, self.gcn_out, K=2)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        if use_positional_encoding:
            self.transformer = Transformer_with_PE(input_dim=self.in_dim + self.gcn_out, hidden_dim=self.hid_dim, num_heads=2, num_layers=2)
        else:
            self.transformer = Transformer_with_PE(input_dim=self.in_dim + self.gcn_out, hidden_dim=self.hid_dim, num_heads=2, num_layers=2)

        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        gn = pm25_hist[:, -1] 
        xgcn = torch.cat((gn, feature[:, -1]), dim=-1) # The history of the features at the latest time step

        xn_xgcn = xgcn
        xn_xgcn = xn_xgcn.contiguous()
        x_gcn = xn_xgcn.view(self.batch_size * self.city_num, -1)
        x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index))
        x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)
        #xn_xgcn = xn_xgcn.unsqueeze(1).repeat(1, pm25_hist.size(1), 1, 1)
        # Initialize the initial hidden state hn
        print ("pm25_hist shape:",pm25_hist.shape,"feature shape:",feature.shape )

        hn = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        #xn = pm25_hist[:,-24:,:]
        xn = pm25_hist  # This adds an extra dimension at index 1

        #print (xn.shape,feature[:, self.hist_len:self.hist_len + self.pred_len].shape )
        # Prepare the input data for the Transformer
        x = torch.cat((xn, feature[:, self.hist_len:self.hist_len + self.pred_len]), dim=-1)
    
        # Concatenate GNN output with the input data and pass it through the Transformer
        print("x shape:",x.shape,"xn_gnn shape:",xn_gnn.shape)
        x = torch.cat([x, xn_gnn], dim=-1)
        hn = self.transformer(x)

        # Get predictions for all time steps
        pm25_pred = self.fc_out(hn)

        return pm25_pred
