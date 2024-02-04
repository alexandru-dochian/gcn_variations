import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import GCNLayer, GPoolBlock, GUnpoolBlock, Normalize, SelfLoops


class GCN(nn.Module):
    def __init__(
        self,
            num_nodes: int,
            in_features: int,
            output_classes: int,
            hidden_features: int,
            gcn_drop_prob: int,
            train_self_loops: bool
    ):
        super().__init__()
        self.normalize = Normalize()
        self.self_loops = SelfLoops(num_nodes, trainable=train_self_loops)
        self.layer_1 = GCNLayer(in_features, hidden_features, gcn_drop_prob)
        self.layer_2 = GCNLayer(hidden_features, output_classes, gcn_drop_prob)

    def forward(self, H, A) -> torch.Tensor:
        A = self.self_loops(A)
        A = self.normalize(A)
        H = self.layer_1(H, A)
        H = self.layer_2(H, A)
        return F.log_softmax(H, dim=1)


class GraphUnet(nn.Module):
    def __init__(
        self,
            num_nodes: int,
            in_features: int,
            output_classes: int,
            k_values: list[int],
            number_of_blocks: int,
            hidden_features: int,
            gcn_drop_prob: int,
            pool_drop_prob: int,
            train_self_loops: bool
    ):
        super().__init__()
        self.k_values = k_values
        self.number_of_blocks = number_of_blocks
        self.depth = len(self.k_values)
        self.normalize = Normalize()
        self.self_loops = SelfLoops(num_nodes, trainable=train_self_loops)
        self.first_gcn = GCNLayer(in_features, hidden_features, gcn_drop_prob)
        self.last_gcn = GCNLayer(
            hidden_features, output_classes, gcn_drop_prob)

        self.pool_blocks = []
        self.unpool_blocks = []
        for _ in range(self.number_of_blocks):
            pools = []
            unpools = []
            for k in self.k_values:
                pooling_size = max(2, int(k*num_nodes))
                pools.append(
                    GPoolBlock(
                        pooling_size,
                        hidden_features,
                        hidden_features,
                        gcn_drop_prob,
                        pool_drop_prob
                    )
                )
                unpools.append(
                    GUnpoolBlock(
                        hidden_features,
                        hidden_features,
                        gcn_drop_prob
                    )
                )
            self.pool_blocks.append(pools)
            self.unpool_blocks.append(unpools)

    def forward(self, H, A) -> torch.Tensor:
        A = self.normalize(A)
        A = self.self_loops(A)

        # First GCN
        H = self.first_gcn(H, A)

        # Pooling
        link_information = []
        H_old_history = []
        A_old_history = []

        for block_index in range(self.number_of_blocks):
            for depth_index in range(self.depth):
                A_old_history.append(A)
                H_old_history.append(H)
                H, A, idx = self.pool_blocks[block_index][depth_index](H, A)
                link_information.append(idx)

            # Unpooling
            for depth_index in range(self.depth):
                # modify it so it matches its corresponding pooling block
                depth_index = self.depth - 1 - depth_index
                H_old = H_old_history[depth_index]
                A_old = A_old_history[depth_index]
                idx = link_information[depth_index]
                H, A = self.unpool_blocks[block_index][depth_index](
                    H, H_old, A_old, idx
                )

        # Last GCN
        H = self.last_gcn(H, A)

        return F.log_softmax(H, dim=1)
