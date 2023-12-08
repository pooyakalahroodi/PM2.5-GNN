import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from transformer_module import Transformer_with_PE  # Import the Transformer class

class GC_Transformer(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, num_heads=4, num_layers=2, dropout=0.1):
        super(GC_Transformer, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32  # Define the hidden dimension size
        self.out_dim = 1
        self.gcn_out = 1

        # Graph Convolution layer
        self.conv = ChebConv(self.in_dim, self.gcn_out, K=2)

        # Transformer module with Positional Encoding
        self.transformer = Transformer_with_PE(input_dim=self.in_dim + self.gcn_out, hidden_dim=self.hid_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

        # Final output layer
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        # Prepare edge index for batch processing
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_index = self.edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1) * city_num
        self.edge_index = self.edge_index.view(2, -1)

    def forward(self, pm25_hist, feature):
        self.edge_index = self.edge_index.to(self.device)
        pm25_pred = []
        for i in range(self.pred_len):
            x = torch.cat((pm25_hist[:, i], feature[:, self.hist_len + i]), dim=-1)

            # Apply GCN
            x_gcn = x.contiguous().view(self.batch_size * self.city_num, -1)
            x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)

            # Combine GCN output with the original input
            x = torch.cat((x, x_gcn), dim=-1)

            # Apply Transformer
            x = x.transpose(0, 1)  # Transformer expects seq_len, batch, features
            x = self.transformer(x)
            x = x.transpose(0, 1)

            # Predict pm25 values
            xn = self.fc_out(x)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred
