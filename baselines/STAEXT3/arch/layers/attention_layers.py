import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

# ==================================================================================
# 1. 重构的通用外部注意力模块
# ==================================================================================
class ExternalAttention(nn.Module):
    """
    通用外部注意力模块 (General External Attention Module)
    此模块现在接收预先初始化的 mk 和 mv 线性层，以实现跨层参数共享。

    参数:
        mk (nn.Module): 预先初始化的线性层，用于将输入查询映射到记忆单元 M_k。
        mv (nn.Module): 预先初始化的线性层，用于从更新后的记忆单元 M_v 生成输出。
    """

    def __init__(self, mk, mv):
        super().__init__()
        # 直接使用外部传入的、共享的记忆单元线性层
        self.mk = mk
        self.mv = mv
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries):
        # queries: (batch, ..., d_model)
        # T: (batch, N, T, d_model)
        # S: (batch, T, N, d_model)

        # 1. 计算注意力图
        attn = self.mk(queries)
        # T: -> (batch, N, T, T)
        # T: -> (batch, T, N, N)

        # 2. 归一化 (在最后一个维度上)
        attn = self.softmax(attn)  # -> (batch, ..., T(或N))

        # 论文中提到的双重归一化 (Double-Normalization) 的第二部分
        # 它在节点/时间步维度上进行归一化，与Softmax形成互补。
        attn = attn / (1e-9 + attn.sum(dim=-2, keepdim=True))

        # 3. 生成输出
        out = self.mv(attn)  # -> (batch, ..., d_model)
        return out


# ==================================================================================
# 2. 重构的外部注意力封装层
# ==================================================================================
class TemporalExternalAttnLayer(nn.Module):
    """
    封装后的时间外部注意力层。
    它接收共享的 mk 和 mv 模块，并将其传递给 ExternalAttention 实例。

    参数:
        model_dim (int): 模型的特征维度 D。
        in_steps (int): 输入时间步长 T。
        feed_forward_dim (int): 前馈网络的隐藏层维度。
        dropout (float): Dropout 比率。
        shared_mk (nn.Module): 所有时间注意力层共享的 M_k 线性层。
        shared_mv (nn.Module): 所有时间注意力层共享的 M_v 线性层。
    """

    def __init__(self, model_dim, in_steps, feed_forward_dim=256, dropout=0.1, shared_mk=None, shared_mv=None):
        super().__init__()
        # 使用传入的共享 mk, mv 初始化外部注意力模块
        self.attn = ExternalAttention(mk=shared_mk, mv=shared_mv)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, N, D)

        # --- 1. Attention Sub-layer with Pre-LN ---
        residual = x
        x_norm = self.norm1(x)  # Pre-Normalization

        # Reshape for temporal attention
        x_permuted = x_norm.permute(0, 2, 1, 3)
        b, n, t, d = x_permuted.shape
        x_reshaped = x_permuted.reshape(b * n, t, d)

        # Apply attention
        out_attn = self.attn(x_reshaped)

        # Restore shape
        out_attn_permuted = out_attn.view(b, n, t, d)
        out_attn_final = out_attn_permuted.permute(0, 2, 1, 3)

        # Add residual connection
        x = residual + self.dropout1(out_attn_final)

        # --- 2. Feed-Forward Sub-layer with Pre-LN ---
        residual = x
        x_norm = self.norm2(x)  # Pre-Normalization

        # Apply feed-forward network
        out_ffn = self.feed_forward(x_norm)

        # Add residual connection
        x = residual + self.dropout2(out_ffn)

        return x


class SpatialExternalAttnLayer(nn.Module):
    """
    封装后的空间外部注意力层。
    它接收共享的 mk 和 mv 模块，并将其传递给 ExternalAttention 实例。

    参数:
        model_dim (int): 模型的特征维度 D。
        num_nodes (int): 空间节点数量 N。
        feed_forward_dim (int): 前馈网络的隐藏层维度。
        dropout (float): Dropout 比率。
        shared_mk (nn.Module): 所有空间注意力层共享的 M_k 线性层。
        shared_mv (nn.Module): 所有空间注意力层共享的 M_v 线性层。
    """

    def __init__(self, model_dim, num_nodes, feed_forward_dim=256, dropout=0.1, shared_mk=None, shared_mv=None):
        super().__init__()
        # 使用传入的共享 mk, mv 初始化外部注意力模块
        self.attn = ExternalAttention(mk=shared_mk, mv=shared_mv)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, N, D)

        # --- 1. Attention Sub-layer with Pre-LN ---
        residual = x
        x_norm = self.norm1(x)  # Pre-Normalization

        # Reshape for spatial attention
        b, t, n, d = x_norm.shape
        x_reshaped = x_norm.reshape(b * t, n, d)

        # Apply attention
        out_attn = self.attn(x_reshaped)

        # Restore shape
        out_attn_final = out_attn.view(b, t, n, d)

        # Add residual connection
        x = residual + self.dropout1(out_attn_final)

        # --- 2. Feed-Forward Sub-layer with Pre-LN ---
        residual = x
        x_norm = self.norm2(x)  # Pre-Normalization

        # Apply feed-forward network
        out_ffn = self.feed_forward(x_norm)

        # Add residual connection
        x = residual + self.dropout2(out_ffn)

        return x
