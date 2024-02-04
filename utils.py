import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def create_A(edge_index: torch.Tensor):
    # assuming edge_index has shape (2, num_edges)
    assert edge_index.shape[0] == 2

    num_nodes = torch.max(edge_index).item() + 1

    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1.0

    return A


def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    predicted_labels = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_labels == y_true).sum().item()
    accuracy = correct_predictions / len(y_true)
    return accuracy


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true: torch.Tensor = y_true.cpu().detach()
    y_pred: torch.Tensor = y_pred.cpu().detach()

    predicted_labels = torch.argmax(y_pred, dim=1).numpy()

    precision = precision_score(
        y_true.numpy(), predicted_labels, average='weighted')
    recall = recall_score(y_true.numpy(), predicted_labels, average='weighted')
    f1 = f1_score(y_true.numpy(), predicted_labels, average='weighted')
    accuracy = accuracy_score(y_true.numpy(), predicted_labels)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

    return metrics


def compute_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    loss_function = torch.nn.NLLLoss()

    # target to torch.long
    return loss_function(y_pred, y_true.to(dtype=torch.long))
