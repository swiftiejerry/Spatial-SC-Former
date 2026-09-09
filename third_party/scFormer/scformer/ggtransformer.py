import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import softmax as Softmax
import math
from .conv import *


class HGTTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, num_layers, dropout=0.2, use_norm=True):
        super(HGTTransformer, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types

        # Add type embeddings
        self.type_embeddings = nn.Embedding(num_types, in_dim)

        # Initialize layer components
        self.hgt_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pre_norm_layers = nn.ModuleList()  # Add pre-normalization

        # Initialize multiple layers
        for _ in range(num_layers):
            self.hgt_layers.append(
                HGTConv(in_dim, out_dim, num_types, num_relations, n_heads, dropout, use_norm)
            )
            self.attention_layers.append(
                MultiHeadSelfAttention(out_dim, n_heads, dropout)
            )
            self.feedforward_layers.append(nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 4, out_dim)
            ))
            self.norm_layers.append(nn.LayerNorm(out_dim))
            self.pre_norm_layers.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(out_dim)

    def forward(self, x, node_type, edge_index, edge_type):
        # Handle list input for heterogeneous features
        if isinstance(x, list):
            # Assuming x[0] is cell features and x[1] is gene features
            x = torch.cat(x, dim=0)

        # Add type embeddings
        type_embed = self.type_embeddings(node_type)
        x = x + type_embed

        h = x
        for i in range(self.num_layers):
            # Pre-norm
            h_norm = self.pre_norm_layers[i](h)

            # HGT convolution
            h_conv = self.hgt_layers[i](h_norm, node_type, edge_index, edge_type)

            # Reshape for attention if needed
            if h_conv.dim() == 2:
                h_conv = h_conv.unsqueeze(0)  # [num_nodes, dim] -> [1, num_nodes, dim]

            # Self-attention
            h_attn = self.attention_layers[i](h_conv)

            # Remove batch dimension if added
            if h_attn.dim() == 3 and h_attn.size(0) == 1:
                h_attn = h_attn.squeeze(0)

            # First residual connection
            h = h + h_attn

            # Feedforward with pre-norm
            h_norm = self.norm_layers[i](h)
            h_ff = self.feedforward_layers[i](h_norm)

            # Second residual connection
            h = h + h_ff

            # Dropout
            h = self.dropout(h)

        # Final layer norm
        h = self.final_norm(h)

        return h


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Handle unbatched input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [num_nodes, dim] -> [1, num_nodes, dim]

        batch_size, num_nodes, embed_dim = x.size()

        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, nodes, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, embed_dim)
        attn_output = self.out_proj(attn_output)

        # Remove batch dimension if input was unbatched
        if x.size(0) == 1:
            attn_output = attn_output.squeeze(0)

        return attn_output