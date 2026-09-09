from .conv import *
from .utils import *
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import OneHotEncoder


class GNN_from_raw(nn.Module):
    def __init__(
        self,
        in_dim,
        n_hid,
        num_types,
        num_relations,
        n_heads,
        n_layers,
        dropout=0.2,
        conv_name="hgt",
        prev_norm=True,
        last_norm=True,
    ):
        super(GNN_from_raw, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()

        # Initialize MLP weight matrices
        for ti in range(num_types):
            self.embedding1.append(nn.Linear(in_dim[ti], 256))

        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid))

        # Initialize graph convolution layers
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(
                    conv_name,
                    n_hid,
                    n_hid,
                    num_types,
                    num_relations,
                    n_heads,
                    dropout,
                    use_norm=prev_norm,
                )
            )
        self.gcs.append(
            GeneralConv(
                conv_name,
                n_hid,
                n_hid,
                num_types,
                num_relations,
                n_heads,
                dropout,
                use_norm=last_norm,
            )
        )

    def encode(self, x, t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return h1

    def forward(self, node_feature, node_type, edge_index, edge_type):
        node_embedding = []
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id], t_id))

        node_embedding = torch.stack(node_embedding)
        # Initialize result matrix
        res = torch.zeros(node_embedding.size(0), self.n_hid).to(node_feature[0].device)

        # Process each node type
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            # Update result matrix
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding[idx]))

        # Apply dropout to the result matrix
        meta_xs = self.drop(res)
        del res

        # Iterate through graph convolution layers and update result matrix
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)

        return meta_xs


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


class NDR_1(nn.Module):
    def __init__(
        self,
        RNA_matrix,
        indices,
        ini_p1,
        n_hid,
        n_heads,
        n_layers,
        labsm,
        lr,
        wd,
        device,
        num_types=2,
        num_relations=2,
        epochs=1,
        loss_contrastive_weight=0.001,
    ):
        super(NDR_1, self).__init__()
        self.RNA_matrix = RNA_matrix
        self.indices = indices
        self.ini_p1 = ini_p1
        self.in_dim = [
            RNA_matrix.shape[0],
            RNA_matrix.shape[1],
        ]  # 仅包含基因和细胞的维度
        self.n_hid = n_hid
        self.num_types = num_types  # 2 种类型：细胞和基因
        self.num_relations = num_relations  # 2 种关系：基因到细胞和细胞到基因
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.labsm = labsm
        self.lr = lr
        self.wd = wd
        self.device = device
        self.epochs = epochs
        self.loss_contrastive_weight = loss_contrastive_weight  # 新增超参数

        # 标签平滑
        self.LabSm = LabelSmoothing(self.labsm)

        # GNN 模型
        self.gnn = GNN_from_raw(
            in_dim=self.in_dim,
            n_hid=self.n_hid,
            num_types=self.num_types,
            num_relations=self.num_relations,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=0.3,
        ).to(self.device)

        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(
            self.gnn.parameters(), lr=self.lr, weight_decay=self.wd
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5, verbose=True
        )

    def supervised_contrastive_loss(
        self, features, labels, batch_labels=None, temperature=0.07
    ):
        """
        Compute supervised contrastive loss with batch labels.

        Args:
            features: tensor of shape [batch_size, embedding_dim]
            labels: tensor of shape [batch_size]
            batch_labels: tensor of shape [batch_size], indicating the batch assignments (optional)
            temperature: a float scalar temperature

        Returns:
            loss: a scalar loss value
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        epsilon = 1e-10  # 防止除以零或对数零

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # Clamp similarity_matrix to prevent large values
        max_sim = 50  # 可以根据需要调整
        min_sim = -50
        similarity_matrix = torch.clamp(similarity_matrix, min=min_sim, max=max_sim)

        # Create masks
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Number of labels does not match number of features")

        # Positive mask: labels match
        label_eq = torch.eq(labels, labels.T).float().to(features.device)

        # If batch labels are provided, ensure positives come from different batches
        if batch_labels is not None:
            batch_labels = batch_labels.contiguous().view(-1, 1)
            batch_neq = (
                (~torch.eq(batch_labels, batch_labels.T)).float().to(features.device)
            )
            mask = label_eq * batch_neq
        else:
            # If no batch labels, use label matching only
            mask = label_eq

        # Remove self-contrast cases
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(features.device)
        mask = mask * logits_mask

        # Compute log_prob
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        exp_sim_sum = exp_sim.sum(1, keepdim=True) + epsilon  # 添加 epsilon 防止 log(0)
        log_prob = similarity_matrix - torch.log(exp_sim_sum)

        # Compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        # 避免除以零
        mean_log_prob_pos = torch.zeros_like(mask_sum)
        non_zero_mask = mask_sum > 0
        mean_log_prob_pos[non_zero_mask] = (
            (mask * log_prob).sum(1)[non_zero_mask]
        ) / mask_sum[non_zero_mask]

        # Loss
        loss = -mean_log_prob_pos
        # Avoid NaNs
        loss = loss[non_zero_mask].mean()
        return loss

    def train_model(self, n_batch, batch_labels_list=None):
        """
        Args:
            n_batch: number of batches
            Node_Ids: array of Node IDs in the current order
            mapping: array indicating the mapping to original order (if shuffled)
            batch_labels_list: (optional) a list where each element is a tensor of batch labels for the cells in that batch
        """
        print(
            "The training process for the NodeDimensionReduction model has started. Please wait."
        )
        h_final = None  # 用于存储最后一个 h

        # 初始化一个数组来存储所有的 cell_emb，按 Node_Ids 顺序
        # total_cells = self.RNA_matrix.shape[1]
        # cell_emb_all = torch.zeros((total_cells, self.n_hid)).to(self.device)

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            # z!!!!!!!
            for batch_id in np.arange(n_batch):
                # 获取当前批次的基因和细胞索引
                gene_index = self.indices[batch_id]["gene_index"]
                cell_index = self.indices[batch_id]["cell_index"]

                # 提取基因和细胞的特征
                gene_feature = self.RNA_matrix[list(gene_index), :]
                cell_feature = self.RNA_matrix[:, list(cell_index)].T

                # 转换为张量并移动到设备上
                gene_feature = torch.tensor(
                    gene_feature.todense(), dtype=torch.float32
                ).to(self.device)
                cell_feature = torch.tensor(
                    cell_feature.todense(), dtype=torch.float32
                ).to(self.device)

                # 构建节点特征列表（细胞和基因）
                node_feature = [cell_feature, gene_feature]

                # 构建基因-细胞子图的邻接矩阵
                gene_cell_sub = self.RNA_matrix[list(gene_index), :][
                    :, list(cell_index)
                ]

                # 构建基因到细胞的边索引
                gene_to_cell_src = list(
                    np.nonzero(gene_cell_sub)[0] + len(cell_index)
                )  # 基因索引从 len(cell_index) 开始
                gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

                # 构建细胞到基因的边索引
                cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
                cell_to_gene_dst = list(
                    np.nonzero(gene_cell_sub)[0] + len(cell_index)
                )  # 基因索引

                # 合并边索引
                edge_index = torch.LongTensor(
                    [
                        gene_to_cell_src + cell_to_gene_src,  # 源节点
                        gene_to_cell_dst + cell_to_gene_dst,  # 目标节点
                    ]
                ).to(self.device)

                # 定义节点类型：0 代表细胞，1 代表基因
                node_type = torch.LongTensor(
                    np.array(
                        list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                    )
                ).to(self.device)

                # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
                edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
                edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
                edge_type = torch.LongTensor(
                    np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
                ).to(self.device)

                # 获取标签
                l = torch.LongTensor(np.array(self.ini_p1)[cell_index]).to(self.device)

                # 获取批次标签
                if batch_labels_list is not None:
                    batch_labels = batch_labels_list[batch_id]
                    batch_labels = torch.LongTensor(batch_labels).to(self.device)
                else:
                    batch_labels = None

                # 前向传播
                node_rep = self.gnn.forward(
                    node_feature, node_type, edge_index, edge_type
                ).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]

                # 将 cell_emb 存储到对应的位置
                # cell_emb_all[cell_index] = cell_emb

                # 解码器：基因与细胞的关系
                decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())
                decoder_cell_to_gene = torch.mm(cell_emb, gene_emb.t())

                # 构建目标矩阵
                gene_cell_sub_tensor = torch.tensor(
                    gene_cell_sub.todense(), dtype=torch.float32
                ).to(self.device)

                # 计算基因到细胞的 KL 散度损失
                logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
                p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction="mean")

                # 计算细胞到基因的 KL 散度损失
                logp_x2 = F.log_softmax(decoder_cell_to_gene, dim=-1)
                p_y2 = F.softmax(gene_cell_sub_tensor.t(), dim=-1)
                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction="mean")

                # 总的 KL 散度损失
                loss_kl = loss_kl1 + loss_kl2

                # 聚类损失
                loss_cluster = self.LabSm(cell_emb, l)

                # 余弦相似度损失
                lll = 0
                g = l.cpu().numpy().tolist()
                unique_labels = set(g)
                for label in unique_labels:
                    mask = np.array(g) == label
                    h = cell_emb[mask]
                    if h.size(0) > 1:
                        # 计算细胞间的余弦相似度
                        similarity = F.cosine_similarity(
                            h.unsqueeze(1), h.unsqueeze(0), dim=-1
                        )
                        lll += similarity.mean()
                        h_final = h  # 更新 h_final 为当前标签的 h

                # 对比学习损失
                loss_contrastive = self.supervised_contrastive_loss(
                    cell_emb, l, batch_labels
                )

                # 总损失
                loss = (
                    loss_cluster
                    + loss_kl
                    + self.loss_contrastive_weight * loss_contrastive
                    - lll
                )
                # loss = loss_cluster + loss_kl - lll

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print("The training for the NodeDimensionReduction model has been completed.")

        return self.gnn, cell_emb, gene_emb, h_final

    def save_model(self, file_path):
        """保存训练好的模型"""
        torch.save(self.gnn.state_dict(), file_path)

    def load_model(self, file_path):
        """加载保存的模型

        Args:
            file_path (str): 模型文件的路径
        """
        # 加载模型的状态字典
        state_dict = torch.load(file_path, map_location=self.device)
        # 将状态字典加载到 GNN 模型中
        self.gnn.load_state_dict(state_dict)
        # 将模型移动到指定的设备
        self.gnn.to(self.device)
        # 设置模型为评估模式（如果需要进行推理）
        self.gnn.eval()
        print(f"Model loaded successfully from {file_path}.")


def ScFormer_pred(RNA_matrix, gnn, indices, device):
    """
    使用训练好的 scformer 模型进行预测。

    :param RNA_matrix: 基因表达矩阵 (scipy.sparse 或类似格式)
    :param gnn: 训练好的 ScFormer模型
    :param indices: 批次索引列表，每个元素包含 'gene_index' 和 'cell_index'
    :param device: 计算设备 (e.g., torch.device('cuda') 或 torch.device('cpu'))
    :param gene_names: 基因名称列表
    :return: 包含预测标签、细胞嵌入以及（如果需要）EGRN 结果的字典
    """
    # n_batch = math.ceil(nodes_id.shape[0] / cell_size)
    n_batch = len(indices)

    embedding = []
    l_pre = []
    ScFormer_result = {}
    with torch.no_grad():
        for batch_id in tqdm(range(n_batch), desc="Prediction Batches"):
            gene_index = indices[batch_id]["gene_index"]
            cell_index = indices[batch_id]["cell_index"]
            # peak_index = indices[batch_id]['peak_index']  # 移除峰值索引

            # 提取基因和细胞的特征
            gene_feature = RNA_matrix[list(gene_index), :]
            cell_feature = RNA_matrix[:, list(cell_index)].T

            # 转换为张量并移动到设备上
            gene_feature = torch.tensor(gene_feature.todense(), dtype=torch.float32).to(
                device
            )
            cell_feature = torch.tensor(cell_feature.todense(), dtype=torch.float32).to(
                device
            )

            # 构建节点特征列表（细胞和基因）
            node_feature = [cell_feature, gene_feature]

            # 构建基因-细胞子图的邻接矩阵
            gene_cell_sub = RNA_matrix[list(gene_index), :][:, list(cell_index)]

            # 构建基因到细胞的边索引
            gene_to_cell_src = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))
            gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

            # 构建细胞到基因的边索引
            cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
            cell_to_gene_dst = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))

            # 合并边索引
            edge_index = torch.LongTensor(
                [
                    gene_to_cell_src + cell_to_gene_src,
                    gene_to_cell_dst + cell_to_gene_dst,
                ]
            ).to(device)

            # 定义节点类型：0 代表细胞，1 代表基因
            node_type = torch.LongTensor(
                np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                )
            ).to(device)

            # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
            edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
            edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
            edge_type = torch.LongTensor(
                np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
            ).to(device)

            # 获取标签
            # l = torch.LongTensor(np.array(nodes_id)[cell_index]).to(device)

            # 前向传播
            node_rep = gnn.forward(node_feature, node_type, edge_index, edge_type).to(
                device
            )
            cell_emb = node_rep[node_type == 0]
            gene_emb = node_rep[node_type == 1]
            # peak_emb = node_rep[node_type == 2]  # 移除峰值嵌入

            # 解码器：基因与细胞的关系
            decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())

            # 构建目标矩阵
            gene_cell_sub_tensor = torch.tensor(
                gene_cell_sub.todense(), dtype=torch.float32
            ).to(device)

            # 计算基因到细胞的 KL 散度损失
            logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
            p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
            loss_kl1 = F.kl_div(logp_x1, p_y1, reduction="mean")

            # 总的 KL 散度损失 (只包括基因-细胞)
            loss_kl = loss_kl1

            # 聚类损失（如果需要）
            # 由于在预测阶段通常不需要计算损失，可以选择移除或保留根据需求
            # 这里只保留嵌入和预测标签

            # 取 cell_emb 的嵌入作为输出
            embedding.append(cell_emb.cpu().numpy())

            # 预测标签为 cell_emb 的某个维度的最大值 (假设分类任务)
            cell_pre = list(cell_emb.argmax(dim=1).detach().cpu().numpy())
            l_pre.extend(cell_pre)

    cell_embedding = np.vstack(embedding)
    cell_clu = np.array(l_pre)

    ScFormer_result = {"pred_label": cell_clu, "cell_embedding": cell_embedding}
    return ScFormer_result


class NDR_2(nn.Module):
    """
    Node Dimension Reduction Model with optional batch correction.

    Args:
        RNA_matrix (np.ndarray): 输入的RNA表达矩阵。
        indices (list): 批次索引信息。
        ini_p1 (list): 初始标签信息。
        n_hid (int): 隐藏层维度。
        n_heads (int): 注意力头数量。
        n_layers (int): GNN层数。
        labsm (float): 标签平滑参数。
        lr (float): 学习率。
        wd (float): 权重衰减。
        device (torch.device): 计算设备。
        enable_batch_correction (bool, optional): 是否启用批次矫正。默认值为False。
        batch_source (list, optional): 批次来源信息。启用批次矫正时必须提供。
        num_types (int, optional): 节点类型数量。默认值为2。
        num_relations (int, optional): 关系类型数量。默认值为2。
        epochs (int, optional): 训练轮数。默认值为1。
    """

    def __init__(
        self,
        RNA_matrix,
        indices,
        ini_p1,
        n_hid,
        n_heads,
        n_layers,
        labsm,
        lr,
        wd,
        device,
        enable_batch_correction=False,
        batch_source=None,
        num_types=2,
        num_relations=2,
        epochs=1,
    ):
        super(NDR_2, self).__init__()
        self.RNA_matrix = RNA_matrix
        self.indices = indices
        self.ini_p1 = ini_p1
        self.in_dim = [
            RNA_matrix.shape[0],
            RNA_matrix.shape[1],
        ]  # 仅包含基因和细胞的维度
        self.n_hid = n_hid
        self.num_types = num_types  # 2 种类型：细胞和基因
        self.num_relations = num_relations  # 2 种关系：基因到细胞和细胞到基因
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.labsm = labsm
        self.lr = lr
        self.wd = wd
        self.device = device
        self.epochs = epochs

        self.enable_batch_correction = enable_batch_correction
        self.batch_source = batch_source

        if self.enable_batch_correction:
            if self.batch_source is None:
                raise ValueError(
                    "batch_source must be provided when enable_batch_correction is True."
                )

            self.batch_encoder = OneHotEncoder(sparse=False)
            self.batch_onehot = self.batch_encoder.fit_transform(
                np.array(self.batch_source).reshape(-1, 1)
            )
            self.n_batches = len(np.unique(self.batch_source))
        else:
            self.batch_encoder = None
            self.batch_onehot = None
            self.n_batches = None

        # 标签平滑
        self.LabSm = LabelSmoothing(self.labsm)

        # GNN 模型
        self.gnn = GNN_from_raw(
            in_dim=self.in_dim,
            n_hid=self.n_hid,
            num_types=self.num_types,
            num_relations=self.num_relations,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=0.3,
        ).to(self.device)

        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(
            self.gnn.parameters(), lr=self.lr, weight_decay=self.wd
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5, verbose=True
        )

        adata = ad.AnnData(self.RNA_matrix.T)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        self.cell_features_processed = csr_matrix(adata.X.T)

    def calculate_batch_loss(self, cell_emb, cell_indices):
        """
        计算批次校正损失

        Args:
            cell_emb: 当前batch中细胞的嵌入表示
            cell_indices: 当前batch中细胞的原始索引
        """
        # 获取当前batch中细胞的批次信息
        current_batch_onehot = torch.FloatTensor(self.batch_onehot[cell_indices]).to(
            self.device
        )

        # 计算成对距离矩阵
        cell_dist = torch.cdist(cell_emb, cell_emb, p=2)

        # 获取相同/不同批次的mask
        batch_similarity = torch.mm(current_batch_onehot, current_batch_onehot.t())
        same_batch_mask = (batch_similarity > 0).float()
        diff_batch_mask = (batch_similarity == 0).float()

        # 计算批次内协方差
        batch_means = []
        batch_vars = []
        for i in range(self.n_batches):
            batch_mask = current_batch_onehot[:, i] == 1
            if torch.sum(batch_mask) > 1:  # 确保至少有两个细胞
                batch_cells = cell_emb[batch_mask]
                batch_means.append(torch.mean(batch_cells, dim=0))
                batch_vars.append(torch.var(batch_cells, dim=0))

        # 计算批次间的分布差异
        if len(batch_means) > 1:
            means_loss = torch.var(torch.stack(batch_means), dim=0).mean()
            vars_loss = torch.var(torch.stack(batch_vars), dim=0).mean()
            distribution_loss = means_loss + 0.5 * vars_loss
        else:
            distribution_loss = torch.tensor(0.0).to(self.device)

        # 计算局部结构保持损失
        neighbor_k = min(15, cell_emb.size(0) - 1)  # 取最近的k个邻居
        dist_sorted, _ = torch.sort(cell_dist, dim=1)
        neighbor_dist = dist_sorted[:, 1 : neighbor_k + 1]  # 排除自身
        structure_loss = torch.mean(neighbor_dist)

        return distribution_loss + 0.5 * structure_loss

    def train_model(self, n_batch):
        """
        Args:
            n_batch: number of batches
        """
        print(
            "The training process for the NodeDimensionReduction model has started. Please wait."
        )
        h_final = None  # 用于存储最后一个 h

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            for batch_id in np.arange(n_batch):
                # 获取当前批次的基因和细胞索引
                gene_index = self.indices[batch_id]["gene_index"]
                cell_index = self.indices[batch_id]["cell_index"]

                # 提取基因和细胞的特征
                gene_feature = self.RNA_matrix[list(gene_index), :]
                cell_feature = self.cell_features_processed[:, list(cell_index)].T

                # 转换为张量并移动到设备上
                gene_feature = torch.tensor(
                    gene_feature.todense(), dtype=torch.float32
                ).to(self.device)
                cell_feature = torch.tensor(
                    cell_feature.todense(), dtype=torch.float32
                ).to(self.device)

                # 构建节点特征列表（细胞和基因）
                node_feature = [cell_feature, gene_feature]

                # 构建基因-细胞子图的邻接矩阵
                gene_cell_sub = self.RNA_matrix[list(gene_index), :][
                    :, list(cell_index)
                ]

                # 构建基因到细胞的边索引
                gene_to_cell_src = list(
                    np.nonzero(gene_cell_sub)[0] + len(cell_index)
                )  # 基因索引从 len(cell_index) 开始
                gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

                # 构建细胞到基因的边索引
                cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
                cell_to_gene_dst = list(
                    np.nonzero(gene_cell_sub)[0] + len(cell_index)
                )  # 基因索引

                # 合并边索引
                edge_index = torch.LongTensor(
                    [
                        gene_to_cell_src + cell_to_gene_src,  # 源节点
                        gene_to_cell_dst + cell_to_gene_dst,  # 目标节点
                    ]
                ).to(self.device)

                # 定义节点类型：0 代表细胞，1 代表基因
                node_type = torch.LongTensor(
                    np.array(
                        list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                    )
                ).to(self.device)

                # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
                edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
                edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
                edge_type = torch.LongTensor(
                    np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
                ).to(self.device)

                # 获取标签
                l = torch.LongTensor(np.array(self.ini_p1)[cell_index]).to(self.device)

                # 前向传播
                node_rep = self.gnn.forward(
                    node_feature, node_type, edge_index, edge_type
                ).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]

                # 解码器：基因与细胞的关系
                decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())
                decoder_cell_to_gene = torch.mm(cell_emb, gene_emb.t())

                # 构建目标矩阵
                gene_cell_sub_tensor = torch.tensor(
                    gene_cell_sub.todense(), dtype=torch.float32
                ).to(self.device)

                # 计算基因到细胞的 KL 散度损失
                logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
                p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction="mean")

                # 计算细胞到基因的 KL 散度损失
                logp_x2 = F.log_softmax(decoder_cell_to_gene, dim=-1)
                p_y2 = F.softmax(gene_cell_sub_tensor.t(), dim=-1)
                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction="mean")

                # 总的 KL 散度损失
                loss_kl = loss_kl1 + loss_kl2

                # 聚类损失
                loss_cluster = self.LabSm(cell_emb, l)

                # 余弦相似度损失
                lll = 0
                g = l.cpu().numpy().tolist()
                unique_labels = set(g)
                for label in unique_labels:
                    mask = np.array(g) == label
                    h = cell_emb[mask]
                    if h.size(0) > 1:
                        # 计算细胞间的余弦相似度
                        similarity = F.cosine_similarity(
                            h.unsqueeze(1), h.unsqueeze(0), dim=-1
                        )
                        lll += similarity.mean()
                        h_final = h  # 更新 h_final 为当前标签的 h

                # 计算总损失
                if self.enable_batch_correction:
                    batch_correction_loss = self.calculate_batch_loss(
                        cell_emb, cell_index
                    )
                    loss = loss_cluster + loss_kl - lll + 0.5 * batch_correction_loss
                else:
                    loss = loss_cluster + loss_kl - lll

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print("The training for the NodeDimensionReduction model has been completed.")

        return self.gnn, cell_emb, gene_emb, h_final

    def save_model(self, file_path):
        """保存训练好的模型"""
        torch.save(self.gnn.state_dict(), file_path)

    def load_model(self, file_path):
        """加载保存的模型

        Args:
            file_path (str): 模型文件的路径
        """
        # 加载模型的状态字典
        state_dict = torch.load(file_path, map_location=self.device)
        # 将状态字典加载到 GNN 模型中
        self.gnn.load_state_dict(state_dict)
        # 将模型移动到指定的设备
        self.gnn.to(self.device)
        # 设置模型为评估模式（如果需要进行推理）
        self.gnn.eval()
        print(f"Model loaded successfully from {file_path}.")


def pred(RNA_matrix, gnn, indices, device):
    """
    使用训练好的 scformer 模型进行预测。

    :param RNA_matrix: 基因表达矩阵 (scipy.sparse 或类似格式)
    :param gnn: 训练好的 ScFormer模型
    :param indices: 批次索引列表，每个元素包含 'gene_index' 和 'cell_index'
    :param device: 计算设备 (e.g., torch.device('cuda') 或 torch.device('cpu'))
    :param gene_names: 基因名称列表
    :return: 包含预测标签、细胞嵌入以及（如果需要）EGRN 结果的字典
    """
    # n_batch = math.ceil(nodes_id.shape[0] / cell_size)
    # 预处理 RNA_matrix
    adata = ad.AnnData(RNA_matrix.T)  # 转置以获取 cell × gene 矩阵
    sc.pp.normalize_total(adata, target_sum=1e4)  # 标准化
    sc.pp.log1p(adata)  # 对数转换
    sc.pp.scale(adata)  # 缩放处理
    RNA_matrix_processed = csr_matrix(adata.X.T)  # 转置回 gene × cell 矩阵
    n_batch = len(indices)

    embedding = []
    l_pre = []
    ScFormer_result = {}
    with torch.no_grad():
        for batch_id in tqdm(range(n_batch), desc="Prediction Batches"):
            gene_index = indices[batch_id]["gene_index"]
            cell_index = indices[batch_id]["cell_index"]
            # peak_index = indices[batch_id]['peak_index']  # 移除峰值索引

            # 提取基因和细胞的特征
            gene_feature = RNA_matrix[list(gene_index), :]
            cell_feature = RNA_matrix_processed[:, list(cell_index)].T

            # 转换为张量并移动到设备上
            gene_feature = torch.tensor(gene_feature.todense(), dtype=torch.float32).to(
                device
            )
            cell_feature = torch.tensor(cell_feature.todense(), dtype=torch.float32).to(
                device
            )

            # 构建节点特征列表（细胞和基因）
            node_feature = [cell_feature, gene_feature]

            # 构建基因-细胞子图的邻接矩阵
            gene_cell_sub = RNA_matrix[list(gene_index), :][:, list(cell_index)]

            # 构建基因到细胞的边索引
            gene_to_cell_src = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))
            gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

            # 构建细胞到基因的边索引
            cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
            cell_to_gene_dst = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))

            # 合并边索引
            edge_index = torch.LongTensor(
                [
                    gene_to_cell_src + cell_to_gene_src,
                    gene_to_cell_dst + cell_to_gene_dst,
                ]
            ).to(device)

            # 定义节点类型：0 代表细胞，1 代表基因
            node_type = torch.LongTensor(
                np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                )
            ).to(device)

            # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
            edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
            edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
            edge_type = torch.LongTensor(
                np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
            ).to(device)

            # 获取标签
            # l = torch.LongTensor(np.array(nodes_id)[cell_index]).to(device)

            # 前向传播
            node_rep = gnn.forward(node_feature, node_type, edge_index, edge_type).to(
                device
            )
            cell_emb = node_rep[node_type == 0]
            gene_emb = node_rep[node_type == 1]
            # peak_emb = node_rep[node_type == 2]  # 移除峰值嵌入

            # 解码器：基因与细胞的关系
            decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())

            # 构建目标矩阵
            gene_cell_sub_tensor = torch.tensor(
                gene_cell_sub.todense(), dtype=torch.float32
            ).to(device)

            # 计算基因到细胞的 KL 散度损失
            logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
            p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
            loss_kl1 = F.kl_div(logp_x1, p_y1, reduction="mean")

            # 总的 KL 散度损失 (只包括基因-细胞)
            loss_kl = loss_kl1

            # 聚类损失（如果需要）
            # 由于在预测阶段通常不需要计算损失，可以选择移除或保留根据需求
            # 这里只保留嵌入和预测标签

            # 取 cell_emb 的嵌入作为输出
            embedding.append(cell_emb.cpu().numpy())

            # 预测标签为 cell_emb 的某个维度的最大值 (假设分类任务)
            cell_pre = list(cell_emb.argmax(dim=1).detach().cpu().numpy())
            l_pre.extend(cell_pre)

    cell_embedding = np.vstack(embedding)
    cell_clu = np.array(l_pre)

    ScFormer_result = {"pred_label": cell_clu, "cell_embedding": cell_embedding}
    return ScFormer_result
