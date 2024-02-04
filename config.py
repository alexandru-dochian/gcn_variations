EXPERIMENTS_CONFIG = [
    {
        "name": "GCN drop 0.2 not trainable self loops",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [],
            "hidden_features": 256,
            "gcn_drop_prob": 0.2,
            "pool_drop_prob": 0,
            "number_of_blocks": 0,
            "train_self_loops": False
        }
    },
    {
        "name": "GCN drop 0.7 not trainable self loops",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [],
            "hidden_features": 256,
            "gcn_drop_prob": 0.7,
            "pool_drop_prob": 0,
            "number_of_blocks": 0,
            "train_self_loops": False
        }
    },
    {
        "name": "GCN drop 0.2 trainable self loops",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [],
            "hidden_features": 256,
            "gcn_drop_prob": 0.2,
            "pool_drop_prob": 0,
            "number_of_blocks": 0,
            "train_self_loops": True
        }
    },
    {
        "name": "GCN drop 0.7 trainable self loops",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [],
            "hidden_features": 256,
            "gcn_drop_prob": 0.7,
            "pool_drop_prob": 0,
            "number_of_blocks": 0,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9] 1 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 1,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.7] 1 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.7],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 1,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.5] 1 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 1,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9, 0.7] 1 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9, 0.7],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 1,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.7, 0.5] 1 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.7, 0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 1,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9, 0.7, 0.5] 1 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9, 0.7, 0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 1,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9, 0.7, 0.5] 2 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9, 0.7, 0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 2,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9, 0.7, 0.5] 3 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9, 0.7, 0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 3,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9, 0.7, 0.5] 4 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9, 0.7, 0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 4,
            "train_self_loops": True
        }
    },
    {
        "name": "Graph U-nets k = [0.9, 0.7, 0.5] 5 block",
        "model": "GraphUnet",
        "dataset": "Cora",
        "model_args": {
            "k_values": [0.9, 0.7, 0.5],
            "hidden_features": 256,
            "gcn_drop_prob": 0.5,
            "pool_drop_prob": 0.1,
            "number_of_blocks": 5,
            "train_self_loops": True
        }
    }
]
