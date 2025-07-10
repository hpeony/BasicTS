import torch
from torch import nn

# 引用同目录下的模块
from .mlp import MultiLayerPerceptron
from .embed import SpatialEmbedding


class DMID(nn.Module):
    """
    模型名称: DMID (Deterministic Manifold IDentity)
    设计哲学:
        1. 继承STID的简洁MLP线性模型结构。
        2. 通过确定性编码和流形学习解决可学习ID嵌入的“不可区分性”理论缺陷。
        3. 显式地将节点的“身份(Identity)”和“相似性(Similarity)”解耦，并提供灵活的重组机制。
        4. 时间嵌入部分与STID完全对齐，确保对比的公平性。
    """

    def __init__(self, **model_args):
        super().__init__()
        # ==================== 1. 参数提取 ====================
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.if_dmid_spatial = model_args.get("if_dmid_spatial", False)
        self.dropout = model_args["dropout"]

        # ==================== 2. 空间嵌入模块 (核心创新点) ====================
        self.node_dim = 0
        if self.if_spatial:
            if self.if_dmid_spatial:
                # 使用我们设计的、解耦的、基于流形的DMID空间嵌入模块
                identity_dim = model_args["identity_dim"]
                similarity_dim = model_args["similarity_dim"]
                combination = model_args["spatial_combination_method"]
                use_manifold = model_args["use_manifold_similarity"]
                self.spatial_embedding_layer = SpatialEmbedding(identity_dim, similarity_dim, self.num_nodes,
                                                                combination, use_manifold)
                # 计算总的空间嵌入维度
                self.node_dim = identity_dim + similarity_dim if combination == 'concat' else similarity_dim
            else:
                # 回退到STID的原始空间嵌入方式 (用于消融实验对比)
                print("Fallback: Using STID's original spatial embedding.")
                self.node_dim = model_args["node_dim_stid"]
                self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
                nn.init.xavier_uniform_(self.node_emb)

        # 将 node_indices 注册为 buffer，避免在 forward 中重复创建，提高效率。
        self.register_buffer('node_indices', torch.arange(self.num_nodes))

        # ==================== 3. 时间嵌入模块 (与STID完全对齐) ====================
        self.temp_dim_tid = 0
        if self.if_time_in_day:
            self.temp_dim_tid = model_args["temp_dim_tid"]
            # 使用可学习的参数，而不是确定性编码
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        self.temp_dim_diw = 0
        if self.if_day_in_week:
            self.temp_dim_diw = model_args["temp_dim_diw"]
            # 使用可学习的参数，而不是确定性编码
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # ==================== 4. 时间序列特征嵌入层 ====================
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # ==================== 5. 编码器 (Encoder) ====================
        # 计算编码器输入的总维度
        self.hidden_dim = self.embed_dim + self.node_dim + self.temp_dim_tid + self.temp_dim_diw
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, self.dropout) for _ in range(self.num_layer)])

        # ==================== 6. 回归层 (Regression Head) ====================
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """DMID的前向传播函数"""
        # 步骤 1: 准备输入
        input_data = history_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = input_data.shape
        device = input_data.device

        # 步骤 2: 时间序列特征嵌入
        input_data_flat = input_data.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1).transpose(1,
                                                                                                            2).unsqueeze(
            -1)
        time_series_emb = self.time_series_emb_layer(input_data_flat)

        # 步骤 3: 准备时空ID嵌入
        # 3.1 空间嵌入
        node_emb_list = []
        if self.if_spatial:
            if self.if_dmid_spatial:
                spatial_embeddings = self.spatial_embedding_layer(self.node_indices)
                node_emb_list.append(
                    spatial_embeddings.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
            else:
                node_emb_list.append(
                    self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # 3.2 时间嵌入 (完全仿照STID的逻辑)
        tem_emb_list = []
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # STID论文中提到，时间戳被归一化到[0, 1]，因此需要乘以时间片总数得到索引。
            time_in_day_indices = (t_i_d_data[:, -1, :] * (self.time_of_day_size - 1)).round().type(
                torch.LongTensor).to(device)
            tid_emb = self.time_in_day_emb[time_in_day_indices]
            tem_emb_list.append(tid_emb.transpose(1, 2).unsqueeze(-1))

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_indices = (d_i_w_data[:, -1, :]).type(torch.LongTensor).to(device)
            diw_emb = self.day_in_week_emb[day_in_week_indices]
            tem_emb_list.append(diw_emb.transpose(1, 2).unsqueeze(-1))

        # 步骤 4: 多源信息融合
        hidden = torch.cat([time_series_emb] + node_emb_list + tem_emb_list, dim=1)

        # 步骤 5: 通过编码器处理融合后的信息
        hidden = self.encoder(hidden)

        # 步骤 6: 通过回归层生成最终预测
        prediction = self.regression_layer(hidden)

        return prediction
