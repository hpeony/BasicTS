import torch
from torch import nn

# 引用同目录下的模块
from .mlp import MultiLayerPerceptron
from .embed import SinusoidalEncoding, SpatialEmbedding

class DMID(nn.Module):
    """
    模型名称: DMID (Deterministic Manifold IDentity)
    设计哲学:
        1. 保持类似STID的简洁线性模型结构。
        2. 解决可学习ID嵌入的“不可区分性”缺陷。
        3. 显式地将节点的“身份”和“相似性”解耦。
        4. 引入确定性编码和流形学习的数学思想。
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

        # ==================== 2. 空间嵌入模块 (核心创新点) ====================
        self.node_dim = 0
        if self.if_spatial:
            if self.if_dmid_spatial:
                identity_dim = model_args["identity_dim"]
                similarity_dim = model_args["similarity_dim"]
                combination = model_args["spatial_combination_method"]
                use_manifold = model_args["use_manifold_similarity"]
                self.spatial_embedding_layer = SpatialEmbedding(identity_dim, similarity_dim, self.num_nodes,
                                                                combination, use_manifold)
                self.node_dim = identity_dim + similarity_dim if combination == 'concat' else similarity_dim
            else:
                print("Using STID's original spatial embedding.")
                self.node_dim = model_args["node_dim_stid"]
                self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
                nn.init.xavier_uniform_(self.node_emb)

        ### <<< MODIFIED START: 性能提升 >>>
        # 将 node_indices 注册为 buffer，避免在 forward 中重复创建，提高效率。
        self.register_buffer('node_indices', torch.arange(self.num_nodes))
        ### <<< MODIFIED END >>>

        # ==================== 3. 时间嵌入模块 ====================
        self.temp_dim_tid = 0
        if self.if_time_in_day:
            self.temp_dim_tid = model_args["temp_dim_tid"]
            self.time_in_day_emb = SinusoidalEncoding(self.temp_dim_tid, self.time_of_day_size)

        self.temp_dim_diw = 0
        if self.if_day_in_week:
            self.temp_dim_diw = model_args["temp_dim_diw"]
            self.day_in_week_emb = SinusoidalEncoding(self.temp_dim_diw, self.day_of_week_size)

        # ==================== 4. 时间序列特征嵌入层 ====================
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # ==================== 5. 编码器 (Encoder) ====================
        self.hidden_dim = self.embed_dim + self.node_dim + self.temp_dim_tid + self.temp_dim_diw
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # ==================== 6. 回归层 (Regression Head) ====================
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """DMID的前向传播函数"""
        input_data = history_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = input_data.shape

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
                # 使用注册为buffer的node_indices
                spatial_embeddings = self.spatial_embedding_layer(self.node_indices)
                node_emb_list.append(
                    spatial_embeddings.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
            else:
                node_emb_list.append(
                    self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # 3.2 时间嵌入
        tem_emb_list = []
        if self.if_time_in_day:
            ### <<< MODIFIED START: 辅助修正 >>>
            # 将使用最后一个时间点，改为使用整个输入窗口的平均时间特征，以获得更稳定的信号。
            t_i_d_data = history_data[..., 1]
            # time_in_day_indices = (t_i_d_data[:, -1, :]).type(torch.LongTensor).to(device)
            time_in_day_indices = t_i_d_data.mean(dim=1).type(torch.LongTensor)
            ### <<< MODIFIED END >>>
            tid_emb = self.time_in_day_emb(time_in_day_indices)
            tem_emb_list.append(tid_emb.transpose(1, 2).unsqueeze(-1))

        if self.if_day_in_week:
            ### <<< MODIFIED START: 辅助修正 >>>
            d_i_w_data = history_data[..., 2]
            # day_in_week_indices = (d_i_w_data[:, -1, :]).type(torch.LongTensor).to(device)
            day_in_week_indices = d_i_w_data.mean(dim=1).type(torch.LongTensor)
            ### <<< MODIFIED END >>>
            diw_emb = self.day_in_week_emb(day_in_week_indices)
            tem_emb_list.append(diw_emb.transpose(1, 2).unsqueeze(-1))

        # 步骤 4: 多源信息融合
        hidden = torch.cat([time_series_emb] + node_emb_list + tem_emb_list, dim=1)

        # 步骤 5: 通过编码器处理融合后的信息
        hidden = self.encoder(hidden)

        # 步骤 6: 通过回归层生成最终预测
        prediction = self.regression_layer(hidden)

        return prediction
