import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.default_config import CONFIG_SET1, set_test_config
from utils.umap_utils import build_knn_graph


def load_data(dataset_name="mnist_test", number=10000):
    from data.toy_data import loading_toy_data

    X, y = loading_toy_data(dataset_name, number=number)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)
    X = F.normalize(X, p=2, dim=1)
    return X, y


def run_maple(X, y, n_neighbors=None):
    from maple.training.training import training_with_bt_dynamic_knn
    from maple.training.ce_umap import ce_umap_original

    if n_neighbors is None:
        n_neighbors = CONFIG_SET1["n_neighbors"]
    elif n_neighbors != CONFIG_SET1["n_neighbors"]:
        set_test_config({"n_neighbors": n_neighbors})

    knn_indices, _, _ = build_knn_graph(X, n_neighbors)
    ego_emb, _, fuzzy_set, _, _, _ = training_with_bt_dynamic_knn(X, knn_indices)
    Z_umap = ce_umap_original(ego_emb, fuzzy_set)
    return Z_umap, y


if __name__ == "__main__":
    X, y = load_data()
    Z_umap, y = run_maple(X, y)
    print(f"embedding shape: {Z_umap.shape}")
