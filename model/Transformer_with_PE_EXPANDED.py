import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:max_len, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(2), :].unsqueeze(0).unsqueeze(0)
        return x

class Transformer_with_PE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=2, dropout=0.2):
        super(Transformer_with_PE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Linear layer to expand the input dimension
        self.expand_dim = nn.Linear(input_dim, input_dim * 3)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=input_dim)

        # Multi-Head Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim*3, num_heads=num_heads, dropout=dropout)

        # Layer Normalization after self-attention
        self.norm1 = nn.LayerNorm(input_dim)

        # Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer Normalization after feedforward
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Expand input dimension
        x1 = self.expand_dim(x)
        print("x1.shape",x1.shape)
        # Apply Positional Encoding
        x = self.positional_encoding(x1)

        # Multi-Head Self-Attention
        batch_size, seq_len, num_cities, embedding_dim = x.size()
        x_flat = x.view(batch_size * seq_len, num_cities, embedding_dim*3)
        attn_output, _ = self.self_attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.view(batch_size, seq_len, num_cities, embedding_dim*3)

        # Residual Connection and Layer Normalization
        x = self.norm1(x + attn_output)

        # Feedforward Layer
        ff_output = self.feedforward(x)

        # Residual Connection and Layer Normalization
        x = self.norm2(x + ff_output)

        return x