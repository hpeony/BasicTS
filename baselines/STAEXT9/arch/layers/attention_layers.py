import torch
import torch.nn as nn


# ==================================================================================
# 1. 从 MEAformer 中借鉴的核心外部注意力模块
# ==================================================================================
class TemporalExternalAttn(nn.Module):
    """
    MEAformer's Temporal External Attention.
    It operates on the temporal dimension.
    """

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):
        # queries: (batch_size * model_dim, num_nodes, in_steps)
        attn = self.mk(queries)  # (batch_size * model_dim, num_nodes, S)
        attn = self.softmax(attn)  # Softmax over num_nodes dimension
        out = self.mv(attn)  # (batch_size * model_dim, num_nodes, in_steps)
        return out


class SpatialExternalAttn(nn.Module):
    """
    Spatial External Attention, adapted for the spatial dimension.
    It operates on the spatial (num_nodes) dimension.
    """

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):
        # queries: (batch_size * in_steps, model_dim, num_nodes)
        attn = self.mk(queries)  # (batch_size * in_steps, model_dim, S)
        attn = self.softmax(attn)  # Softmax over model_dim dimension
        out = self.mv(attn)  # (batch_size * in_steps, model_dim, num_nodes)
        return out


# ==================================================================================
# 2. 完整的外部注意力层封装 (类似 LEncoderLayer 和 SelfAttentionLayer)
# ==================================================================================
class TemporalExternalAttnLayer(nn.Module):
    """
    A complete layer for Temporal External Attention, including residual connections,
    layer normalization, and a feed-forward network.
    """

    def __init__(self, model_dim, in_steps, num_nodes, feed_forward_dim=256, dropout=0.1, ext_mem_S=64):
        super().__init__()

        self.attn = TemporalExternalAttn(in_steps, S=ext_mem_S)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=1):
        # x shape: (batch_size, in_steps, num_nodes, model_dim)
        # The `dim` argument is kept for API consistency but is fixed to temporal attention.
        # --- Sub-layer 1: Temporal External Attention ---
        residual = x

        # Permute for attention: (B, T, N, D) -> (B, D, N, T)
        x_permuted = x.permute(0, 3, 2, 1)
        b, d, n, t = x_permuted.shape

        # Reshape: (B, D, N, T) -> (B*D, N, T)
        x_reshaped = x_permuted.reshape(b * d, n, t)

        # Apply external attention
        out_attn = self.attn(x_reshaped)

        # Reshape back and permute to original view
        out_attn_permuted = out_attn.view(b, d, n, t)
        out_attn_final = out_attn_permuted.permute(0, 3, 2, 1)

        # Dropout, Residual, and Norm
        x = self.dropout1(out_attn_final)
        x = self.norm1(x + residual)

        # --- Sub-layer 2: Feed-Forward Network (acting on model_dim) ---
        residual = x
        out_ffn = self.feed_forward(x)
        x = self.dropout2(out_ffn)
        x = self.norm2(x + residual)

        return x


class SpatialExternalAttnLayer(nn.Module):
    """
    A complete layer for Spatial External Attention, including residual connections,
    layer normalization, and a feed-forward network.
    """

    def __init__(self, model_dim, in_steps, num_nodes, feed_forward_dim=256, dropout=0.1, ext_mem_S=64):
        super().__init__()

        self.attn = SpatialExternalAttn(num_nodes, S=ext_mem_S)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=2):
        # x shape: (batch_size, in_steps, num_nodes, model_dim)
        # The `dim` argument is kept for API consistency but is fixed to spatial attention.

        # --- Sub-layer 1: Spatial External Attention ---
        residual = x

        # Permute for attention: (B, T, N, D) -> (B, T, D, N)
        x_permuted = x.permute(0, 1, 3, 2)
        b, t, d, n = x_permuted.shape

        # Reshape: (B, T, D, N) -> (B*T, D, N)
        x_reshaped = x_permuted.reshape(b * t, d, n)

        # Apply external attention
        out_attn = self.attn(x_reshaped)

        # Reshape back and permute to original view
        out_attn_permuted = out_attn.view(b, t, d, n)
        out_attn_final = out_attn_permuted.permute(0, 1, 3, 2)

        # Dropout, Residual, and Norm
        x = self.dropout1(out_attn_final)
        x = self.norm1(x + residual)

        # --- Sub-layer 2: Feed-Forward Network (acting on model_dim) ---
        residual = x
        out_ffn = self.feed_forward(x)
        x = self.dropout2(out_ffn)
        x = self.norm2(x + residual)

        return x
