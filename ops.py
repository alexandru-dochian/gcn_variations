import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def forward(self, A: torch.Tensor):
        degree = torch.diag(torch.sum(A, dim=1))

        # convert degree to torch.float32 for inverse to work
        degree_hat = torch.sqrt(
            # degree
            torch.inverse(
                degree.to(device=A.device, dtype=torch.float32)
            )
        ).to(device=A.device, dtype=A.dtype)

        return degree_hat @ A @ degree_hat


class SelfLoops(nn.Module):
    def __init__(self, num_nodes: int, trainable: bool, initial_factor: int = 2):
        super().__init__()
        self.mask = torch.ones(num_nodes) * initial_factor

        if trainable:
            self.mask = nn.Parameter(self.mask)

    def forward(self, A: torch.Tensor):
        # ReLU ensures non-negativity during backpropagation
        activated_mask = F.relu(self.mask)
        dense_mask = torch.diag(activated_mask)
        return A + dense_mask.to(device=A.device, dtype=A.dtype)


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, drop_prob: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.Tensor(in_features, out_features)
        )
        nn.init.xavier_uniform_(self.weight)
        self.drop = nn.Dropout(p=drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, H: torch.Tensor, A: torch.Tensor):
        H = self.drop(H)
        H = A @ H @ self.weight.to(device=H.device, dtype=H.dtype)
        H = F.relu(H)
        return H

###############################################################################
################### GRAPH POOLING #############################################
###############################################################################


class TopKGraph(nn.Module):
    def forward(self, scores, H, A, pooling_size):
        values, idx = torch.topk(scores, pooling_size)
        values = torch.unsqueeze(values, -1)

        pooled_H = H[idx, :]
        pooled_H = torch.mul(pooled_H, values)

        pooled_A = A[idx, :][:, idx]
        return pooled_H, pooled_A, idx


class Pool(nn.Module):
    def __init__(self, pooling_size, in_features, drop_prob):
        super().__init__()
        self.pooling_size = pooling_size
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_features, 1)
        self.drop = nn.Dropout(p=drop_prob) if drop_prob > 0 else nn.Identity()
        self.top_k_graph = TopKGraph()

    def forward(self, H, A):
        self.proj = self.proj.to(device=H.device, dtype=H.dtype)
        Z = self.drop(H)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return self.top_k_graph(scores, H, A, self.pooling_size)


class GPoolBlock(nn.Module):
    def __init__(
        self,
        pooling_size,
        in_features,
        out_features,
        gcn_drop_prob,
        pool_drop_prob,
    ):
        super().__init__()
        self.gcn_layer = GCNLayer(in_features, out_features, gcn_drop_prob)
        self.pool_layer = Pool(pooling_size, out_features, pool_drop_prob)

    def forward(self, H, A):
        H, A, idx = self.pool_layer(H, A)
        H = self.gcn_layer(H, A)
        return H, A, idx


###############################################################################
################### GRAPH UNPOOLING ###########################################
###############################################################################


class Unpool(nn.Module):
    def forward(self, pooled_H, A_old, idx):
        initial_num_nodes = A_old.shape[0]
        num_features = pooled_H.shape[1]

        H = torch.zeros(
            [initial_num_nodes, num_features],
            device=pooled_H.device
        )
        H[idx] = pooled_H

        return H, A_old


class GUnpoolBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        gcn_drop_prob,
    ):
        super().__init__()
        self.gcn_layer = GCNLayer(in_features, out_features, gcn_drop_prob)
        self.unpool_layer = Unpool()

    def forward(self, H, H_old, A_old, idx):
        H, A = self.unpool_layer(H, A_old, idx)
        H = self.gcn_layer(H + H_old, A)
        return H, A
