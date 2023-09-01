import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, dropout=0.1):
        super(Transformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-Head Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout)

        # Layer Normalization after self-attention
        self.norm1 = nn.LayerNorm(input_size)

        # Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        # Layer Normalization after feedforward
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output, _ = self.self_attention(x, x, x)

        # Residual Connection and Layer Normalization
        x = self.norm1(x + attn_output)

        # Feedforward Layer
        ff_output = self.feedforward(x)

        # Residual Connection and Layer Normalization
        x = self.norm2(x + ff_output)

        return x
