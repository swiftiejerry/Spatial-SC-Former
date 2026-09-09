import anndata as ad
import torch
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import math
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed, cpu_count
import pickle
import os


def subgraph(graph, seed, n_neighbors, node_sele_prob):
    """
    Generates a subgraph starting from a seed node.

    Parameters:
    - graph (scipy.sparse matrix): Adjacency matrix of the graph.
    - seed (int): Seed node ID.
    - n_neighbors (list of int): Number of neighbors to select at each layer.
    - node_sele_prob (np.array): Selection probabilities for nodes.

    Returns:
    - List of selected node indices excluding the seed.
    """
    picked_nodes = {seed}
    last_layer_nodes = {seed}

    # Iteratively select neighbors for each layer
    for layer, n_neighbors_current in enumerate(n_neighbors):
        # Find all neighbors of the last layer nodes
        neighbors = graph[list(last_layer_nodes), :].nonzero()[1]
        neighbors = np.unique(neighbors)

        # If there are no neighbors, terminate early
        if len(neighbors) == 0:
            break

        # Get selection probabilities for the neighbors
        neighbors_prob = node_sele_prob[neighbors]
        neighbors_prob = softmax_stable(neighbors_prob)  # Normalize probabilities

        # Determine the number of neighbors to pick
        to_pick = n_neighbors_current
        n_neighbors_real = min(to_pick, len(neighbors))

        # Select neighbors without replacement based on probabilities
        selected_neighbors = np.random.choice(
            neighbors, size=n_neighbors_real, replace=False, p=neighbors_prob
        )

        # Update the sets of picked and last layer nodes
        last_layer_nodes = set(selected_neighbors)
        picked_nodes.update(last_layer_nodes)

    # Exclude the seed node from the final indices
    indices = sorted(picked_nodes - {seed})
    return indices


def batch_select_whole(
    RNA_matrix, neighbor=[20], cell_size=30, save_path="processed_data_subset"
):
    """
    修改后的函数，支持数据的保存和并行处理。
    """
    # 检查是否已经存在处理后的数据
    indices_ss_file = os.path.join(save_path, "indices_ss.pkl")
    Node_Ids_file = os.path.join(save_path, "Node_Ids.pkl")
    dic_file = os.path.join(save_path, "dic.pkl")

    if (
        os.path.exists(indices_ss_file)
        and os.path.exists(Node_Ids_file)
        and os.path.exists(dic_file)
    ):
        print("正在从磁盘加载处理后的数据...")
        with open(indices_ss_file, "rb") as f:
            indices_ss = pickle.load(f)
        with open(Node_Ids_file, "rb") as f:
            Node_Ids = pickle.load(f)
        with open(dic_file, "rb") as f:
            dic = pickle.load(f)
    else:
        print("正在将数据划分为批次并进行并行处理。请稍候...")

        # 打乱细胞 ID
        Node_Ids = np.random.choice(
            RNA_matrix.shape[1], size=RNA_matrix.shape[1], replace=False
        )
        n_batch = math.ceil(Node_Ids.shape[0] / cell_size)
        indices_ss = []

        if issparse(RNA_matrix):
            RNA_matrix = RNA_matrix.tocsr()  # 确保行切片效率高
        else:
            RNA_matrix = csr_matrix(RNA_matrix)

        dic = {}

        for i in tqdm(range(n_batch), desc="处理批次"):
            gene_indices_all = []
            cell_indices_all = []

            # 确定当前批次的范围
            start_idx = i * cell_size
            end_idx = min((i + 1) * cell_size, Node_Ids.shape[0])
            batch_node_ids = Node_Ids[start_idx:end_idx]

            # 并行处理每个节点
            # results = Parallel(n_jobs=max(1, cpu_count() // 2))(
            #     delayed(process_node)(node, RNA_matrix, neighbor)
            #     for node in batch_node_ids
            # )
            results = Parallel(n_jobs=1)(
                delayed(process_node)(node, RNA_matrix, neighbor)
                for node in batch_node_ids
            )

            for node, gene_indices in results:
                dic[node] = {"g": gene_indices}
                gene_indices_all.extend(gene_indices)

            # 移除重复的基因索引
            gene_indices_all = sorted(set(gene_indices_all))

            # 准备批次字典
            batch_dict = {
                "gene_index": gene_indices_all,
                "cell_index": list(batch_node_ids),
            }
            indices_ss.append(batch_dict)

        # 保存处理后的数据
        os.makedirs(save_path, exist_ok=True)
        with open(indices_ss_file, "wb") as f:
            pickle.dump(indices_ss, f)
        with open(Node_Ids_file, "wb") as f:
            pickle.dump(Node_Ids, f)
        with open(dic_file, "wb") as f:
            pickle.dump(dic, f)

    return indices_ss, Node_Ids, dic


def process_node(node, RNA_matrix, neighbor):
    # 提取当前细胞的基因表达
    rna_expression = RNA_matrix[:, node].toarray().flatten()
    rna_expression[rna_expression < 5] = 0  # 阈值处理

    # 计算选择概率
    selection_prob = np.log(rna_expression + 1)
    selection_prob = np.squeeze(selection_prob)

    # 为当前细胞生成子图
    gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor, selection_prob)

    return node, gene_indices


def softmax_stable(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_simple(x):
    return np.exp(x) / np.exp(x).sum()


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0, num_classes=10):
        """Constructor for LabelSmoothing module.
        :param smoothing: Label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.num_classes = num_classes

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

    # def update_num_classes(self, new_num_classes):
    #     self.num_classes = new_num_classes


def initial_clustering(
    RNA_matrix, custom_n_neighbors=None, n_pcs=40, custom_resolution=None, use_rep=None
):
    print(
        "\tWhen the number of cells is less than or equal to 500, it is recommended to set the resolution value to 0.2."
    )
    print(
        "\tWhen the number of cells is within the range of 500 to 5000, the resolution value should be set to 0.5."
    )
    print(
        "\tWhen the number of cells is greater than 5000, the resolution value should be set to 0.8."
    )

    def segment_function(x):
        if x <= 500:
            return 0.2, 5
        elif x <= 5000:
            return 0.5, 10
        else:
            return 0.8, 15

    adata = ad.AnnData(RNA_matrix.transpose())

    # If the user did not provide a custom resolution or n_neighbors value, use the values calculated by segment_function
    if custom_resolution is None or custom_n_neighbors is None:
        resolution, n_neighbors = segment_function(adata.shape[0])
    else:
        resolution = custom_resolution
        n_neighbors = custom_n_neighbors

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver="arpack")

    # Use the user-provided embedding if available, otherwise use n_pcs
    if use_rep is not None:
        adata.obsm["use_rep"] = use_rep
        sc.pp.neighbors(adata, use_rep="use_rep", n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    sc.tl.leiden(adata, resolution)
    return adata.obs["leiden"]


def purity_score(y_true, y_pred):
    """Purity score

    Args:
        y_true (np.ndarray): n*1 matrix, true labels
        y_pred (np.ndarray): n*1 matrix, predicted clusters

    Returns:
        float: Purity score
    """
    # Create a matrix to store the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)

    # Sort the labels
    # Some labels might be missing, e.g., a set {0,2} where 1 is missing
    # First, find the unique labels and then map them to an ordered set
    # E.g., {0,2} should be mapped to {0,1}
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    y_true = np.array(y_true, dtype="int64")

    # Update the unique labels
    labels = np.unique(y_true)

    # Set the number of bins to n_classes + 2 so that we can compute the actual
    # class occurrences between two consecutive bins
    # The larger bin is excluded: [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most frequent label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    y_true = np.array(y_true, dtype="int8")
    y_voted_labels = np.array(y_voted_labels, dtype="int8")
    return accuracy_score(y_true, y_voted_labels), y_true


def Entropy(pred_label, true_label):
    e = 0
    for k in set(pred_label):
        en = 0
        pred_k = Counter(pred_label)[k]
        index_pred_k = pred_label == k
        for j in set(true_label):
            true_j = Counter(true_label)[j]
            intersection_kj = (true_label[index_pred_k] == j).sum()
            p = np.array(intersection_kj) / np.array(pred_k)
            if p != 0:
                en += np.log(p) * p
        e = e + en * pred_k / true_label.shape[0]
    return abs(e)


def compute_lisi(data, labels, k=30):
    """
    Compute Local Inverse Simpson's Index (LISI) for each point to evaluate batch mixing.

    Parameters:
    data: numpy.ndarray or pandas.DataFrame
        A 2D array with shape (n_samples, n_features) representing the high-dimensional data.
    labels: list or numpy.ndarray
        A list or array of batch labels corresponding to each sample.
    k: int, optional (default=30)
        Number of neighbors to consider for LISI calculation.

    Returns:
    lisi_scores: list
        A list of LISI scores for each sample, representing the diversity of batches among neighbors.
    """
    # Convert data and labels into numpy arrays if they are not
    if isinstance(data, pd.DataFrame):
        data = data.values
    labels = np.array(labels)

    # Initialize nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(
        data
    )  # k+1 because the point itself is included
    distances, indices = nbrs.kneighbors(data)

    lisi_scores = []
    for i in range(len(data)):
        neighbor_labels = labels[
            indices[i][1:]
        ]  # Ignore the point itself by selecting indices from 1 to k
        label_counts = Counter(neighbor_labels)
        total_count = sum(label_counts.values())
        # Compute Simpson's index (sum of p_i^2) where p_i is the proportion of label i in the neighbors
        simpson_index = sum(
            (count / total_count) ** 2 for count in label_counts.values()
        )
        # Inverse Simpson's index
        lisi = 1.0 / simpson_index
        lisi_scores.append(lisi)

    return lisi_scores


def compute_ilisi(data, labels, k=30):
    """
    Compute Integrated Local Inverse Simpson's Index (iLISI) to evaluate overall batch mixing.

    Parameters:
    data: numpy.ndarray or pandas.DataFrame
        A 2D array with shape (n_samples, n_features) representing the high-dimensional data.
    labels: list or numpy.ndarray
        A list or array of batch labels corresponding to each sample.
    k: int, optional (default=30)
        Number of neighbors to consider for iLISI calculation.

    Returns:
    ilisi_score: float
        The mean iLISI score for the entire dataset, representing the average batch diversity among neighbors.
    """
    lisi_scores = compute_lisi(data, labels, k)
    ilisi_score = np.mean(lisi_scores)
    return ilisi_score


def compute_nmi(true_labels, predicted_labels):
    """
    Compute Normalized Mutual Information (NMI).

    Parameters:
    true_labels: list or numpy.ndarray
        True class labels for each sample.
    predicted_labels: list or numpy.ndarray
        Predicted class labels for each sample.

    Returns:
    nmi_score: float
        The Normalized Mutual Information score.
    """
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
    return nmi_score


def compute_ari(true_labels, predicted_labels):
    """
    Compute Adjusted Rand Index (ARI).

    Parameters:
    true_labels: list or numpy.ndarray
        True class labels for each sample.
    predicted_labels: list or numpy.ndarray
        Predicted class labels for each sample.

    Returns:
    ari_score: float
        The Adjusted Rand Index score.
    """
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    return ari_score


def compute_nn_entropy(data, labels, k=30):
    """
    Compute Nearest-Neighbor Entropy (NN entropy) to evaluate batch mixing.

    Parameters:
    data: numpy.ndarray or pandas.DataFrame
        A 2D array with shape (n_samples, n_features) representing the high-dimensional data.
    labels: list or numpy.ndarray
        A list or array of batch labels corresponding to each sample.
    k: int, optional (default=30)
        Number of neighbors to consider for NN entropy calculation.

    Returns:
    mean_nn_entropy: float
        The mean NN entropy score for the entire dataset, representing the average entropy of batch labels among neighbors.
    """
    # Convert data and labels into numpy arrays if they are not
    if isinstance(data, pd.DataFrame):
        data = data.values
    labels = np.array(labels)

    # Initialize nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(
        data
    )  # k+1 because the point itself is included
    distances, indices = nbrs.kneighbors(data)

    nn_entropy_scores = []
    for i in range(len(data)):
        neighbor_labels = labels[
            indices[i][1:]
        ]  # Ignore the point itself by selecting indices from 1 to k
        label_counts = Counter(neighbor_labels)
        total_count = sum(label_counts.values())
        # Compute entropy (sum of -p_i * log(p_i)) where p_i is the proportion of label i in the neighbors
        entropy = -sum(
            (count / total_count) * math.log(count / total_count)
            for count in label_counts.values()
            if count > 0
        )
        nn_entropy_scores.append(entropy)

    mean_nn_entropy = np.mean(nn_entropy_scores)
    return mean_nn_entropy


def compute_silhouette_score(data, labels):
    """
    Compute Silhouette Coefficient to evaluate clustering quality.

    Parameters:
    data: numpy.ndarray or pandas.DataFrame
        A 2D array with shape (n_samples, n_features) representing the high-dimensional data.
    labels: list or numpy.ndarray
        A list or array of cluster labels corresponding to each sample.

    Returns:
    silhouette: float
        The mean Silhouette Coefficient for all samples.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    silhouette = silhouette_score(data, labels)
    return silhouette


def compute_davies_bouldin_score(data, labels):
    """
    Compute Davies-Bouldin Index to evaluate clustering quality.

    Parameters:
    data: numpy.ndarray or pandas.DataFrame
        A 2D array with shape (n_samples, n_features) representing the high-dimensional data.
    labels: list or numpy.ndarray
        A list or array of cluster labels corresponding to each sample.

    Returns:
    db_index: float
        The Davies-Bouldin Index for the given clustering.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    db_index = davies_bouldin_score(data, labels)
    return db_index


def compute_calinski_harabasz_score(data, labels):
    """
    Compute Calinski-Harabasz Index (CH Index) to evaluate clustering quality.

    Parameters:
    data: numpy.ndarray or pandas.DataFrame
        A 2D array with shape (n_samples, n_features) representing the high-dimensional data.
    labels: list or numpy.ndarray
        A list or array of cluster labels corresponding to each sample.

    Returns:
    ch_index: float
        The Calinski-Harabasz Index for the given clustering.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    ch_index = calinski_harabasz_score(data, labels)
    return ch_index
