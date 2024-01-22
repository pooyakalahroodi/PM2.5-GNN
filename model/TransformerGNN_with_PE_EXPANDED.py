import torch
from torch import nn, permute_copy
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter
from model.Transformer_with_PE import Transformer_with_PE, PositionalEncoding

class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        e_h = 32
        e_out = 30
        n_out = out_dim
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        src_wind = node_src[:,:,-2:] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]

        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)
        edge_weight = edge_weight.to(self.device)
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:,:,None]], dim=-1)

        out = self.edge_mlp(out)
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # For higher version of PyG.

        out = out_add + out_sub
        out = self.node_mlp(out)

        return out


class TransformerGNN_with_PE(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std, use_positional_encoding=True):
        super(TransformerGNN_with_PE, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 42

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)

        if use_positional_encoding:
            self.transformer = Transformer_with_PE(input_dim=self.in_dim + self.gnn_out, hidden_dim=self.hid_dim, num_heads=3, num_layers=2)
        else:
            self.transformer = Transformer_with_PE(input_dim=self.in_dim + self.gnn_out, hidden_dim=self.hid_dim, num_heads=3, num_layers=2)

        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        print("pm history shape : ",pm25_hist[-1].shape,"features shape : ",feature[-1].shape)
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        gn = pm25_hist[:, -1] 
        xgnn = torch.cat((gn, feature[:, -1]), dim=-1) # The history of the features at the latest time step
        print("gn shape : ",gn.shape,"xgnn shape : ",xgnn.shape)

        xn_gnn = xgnn
        xn_gnn = xn_gnn.contiguous()
        xn_gnn = self.graph_gnn(xn_gnn)
        xn_gnn = xn_gnn.unsqueeze(1).repeat(1, pm25_hist.size(1), 1, 1)
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
        print("Shape of hn before fc_out:", hn.shape)

        # Get predictions for all time steps
        pm25_pred = self.fc_out(hn)

        return pm25_pred
