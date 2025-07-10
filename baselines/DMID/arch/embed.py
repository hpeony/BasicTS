import torch
from torch import nn
import math


class SinusoidalEncoding(nn.Module):
    """
    模块功能: 确定性的傅里叶特征(正弦/余弦)编码。
    设计方法: 采用源自"Attention Is All You Need" Transformer论文的傅里叶特征方法。
              它为每个离散的位置/ID生成一个唯一的、固定的、非学习的嵌入向量。
              这种方法能够捕捉序数关系和周期性，并从根本上保证了ID的唯一性，解决了可学习嵌入的“不可区分性”风险。
    """

    def __init__(self, d_model: int, max_len: int):
        """
        功能: 初始化一个固定的、非学习的正弦编码矩阵。
        参数:
            d_model (int): 嵌入向量的维度。
            max_len (int): 序列的最大长度或ID的最大数量。
        """
        super().__init__()
        # 创建一个足够大的位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 创建一个位置索引张量，形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算频率项，这是编码的核心，不同的维度对应不同的波长
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数维度使用sin函数进行编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用cos函数进行编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将pe注册为模型的buffer。buffer是模型状态的一部分，但不是参数(不会被训练)。
        # 这样做可以确保pe会随着模型一起移动到CPU/GPU，并且会被保存在state_dict中。
        self.register_buffer('pe', pe)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        功能: 根据输入的索引查找并返回对应的正弦编码。
        参数:
            indices (torch.Tensor): 包含ID索引的张量。
        返回:
            torch.Tensor: 对应的嵌入向量。
        """
        # 直接从预计算的矩阵中按索引查找
        return self.pe[indices]


class SpatialEmbedding(nn.Module):
    """
    模块功能: DMID的核心空间嵌入模块，精确实现“解耦与重组”。
    设计方法:
              1. E_identity (身份嵌入): 使用确定性的 `SinusoidalEncoding` (傅里叶特征) 保证每个节点的唯一性。
              2. E_similarity (相似性嵌入): 采用流形学习方法，将可学习的向量通过L2归一化投影到单位球面上，
                 以更准确地捕捉节点间的非欧几何关系。
    """

    def __init__(self, identity_dim: int, similarity_dim: int, num_nodes: int, combination: str = 'concat', use_manifold_similarity: bool = True):
        """
        功能: 初始化空间嵌入模块。
        参数:
            identity_dim (int): 确定性身份嵌入的维度。
            similarity_dim (int): 可学习相似性嵌入的维度。
            num_nodes (int): 空间节点（传感器）的总数。
            combination (str): 组合方式, 'concat' 或 'gated_add'。
            use_manifold_similarity (bool): 是否在相似性嵌入上使用流形学习（球面投影）。
        """
        super().__init__()
        self.identity_dim = identity_dim
        self.similarity_dim = similarity_dim
        self.num_nodes = num_nodes
        self.combination = combination
        self.use_manifold_similarity = use_manifold_similarity

        # 身份嵌入 E_identity: 使用确定性的正弦编码（傅里叶特征），不可学习。
        self.identity_embedding = SinusoidalEncoding(d_model=identity_dim, max_len=num_nodes)

        # 原始相似性嵌入 Raw E_similarity: 一个标准的可学习参数，作为投影到流形之前的“原材料”。
        self.raw_similarity_embedding = nn.Parameter(torch.empty(num_nodes, similarity_dim))
        nn.init.xavier_uniform_(self.raw_similarity_embedding)

        # 如果使用门控加法，定义投影层和门控参数。
        if self.combination == 'gated_add':
            self.identity_proj = nn.Linear(identity_dim, similarity_dim) if identity_dim != similarity_dim else nn.Identity()
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        功能: 生成最终的空间嵌入。
        参数:
            node_indices (torch.Tensor): 包含节点索引的张量，例如 torch.arange(0, num_nodes)。
        返回:
            torch.Tensor: 最终的空间嵌入向量，形状 [N, D_final]。
        """
        # 步骤1: 获取确定性身份嵌入 E_identity
        e_identity = self.identity_embedding(node_indices)  # Shape: [N, identity_dim]

        # 步骤2: 获取并（可选地）在流形上处理相似性嵌入 E_similarity
        e_similarity_raw = self.raw_similarity_embedding  # Shape: [N, similarity_dim]

        if self.use_manifold_similarity:
            # --- 应用流形学习：将原始嵌入通过L2归一化投影到单位球面上 ---
            # 这种方法在几何上更严谨，直接对应将向量约束在球面流形上。
            # 加上一个很小的epsilon以防止除以零的错误。
            norm = torch.norm(e_similarity_raw, p=2, dim=1, keepdim=True)
            e_similarity = e_similarity_raw / (norm + 1e-8)
        else:
            # 直接使用欧几里得空间中的原始可学习嵌入
            e_similarity = e_similarity_raw

        # 步骤3: 组合两种嵌入
        if self.combination == 'concat':
            # 直接拼接，保留两种信息的完整性
            e_final = torch.cat([e_identity, e_similarity], dim=-1) # Shape: [N, identity_dim + similarity_dim]
        elif self.combination == 'gated_add':
            # 通过可学习的门控动态融合两种信息
            gate = torch.sigmoid(self.alpha)
            e_identity_proj = self.identity_proj(e_identity)
            e_final = gate * e_identity_proj + (1.0 - gate) * e_similarity # Shape: [N, similarity_dim]
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")

        return e_final
