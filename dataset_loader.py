from torch_geometric.datasets import Planetoid
from torch_geometric.data.data import Data

DATASETS = ["Cora", "Citeseer", "Pubmed"]
datasets_root = "./data"


def load_dataset(dataset_name: str) -> Data:
    assert dataset_name in DATASETS, f"Dataset {dataset_name} not found!"
    # download the dataset
    dataset = Planetoid(root='./data', name=dataset_name)

    return dataset[0]
