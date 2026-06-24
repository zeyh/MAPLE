import torch
import torch.nn.functional as F
import numpy as np

def get_curriculum_weights(epoch, total_epochs, min_alpha=0.0, max_alpha=1.0):
    if False:
        progress = epoch / total_epochs
        alpha = max_alpha - progress * (max_alpha - min_alpha)
        beta = 1.0 - alpha
        return alpha, beta
    else:
        mmcr_epochs = 5 
        bt_epochs = 15  
        
        if epoch < bt_epochs:
            alpha = 0.0
            beta = 1.0
        else:
            alpha = 1.0
            beta = 0.0
        
        return alpha, beta


def build_view2(neighbors, mode="mean", temperature=0.1, anchor=None, knn_dists=None):
    if mode == "mean": 
        return neighbors.mean(dim=1)  

    elif mode == "confidence_mask":
        assert knn_dists is not None, "knn_dists is from default knn_dists on raw X"
        weights = torch.exp(-knn_dists / temperature)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return torch.einsum("bn,bnd->bd", weights, neighbors)

    elif mode == "dynamic_confidence":
        assert anchor is not None
        distances = torch.cdist(anchor.unsqueeze(1), neighbors, p=2).squeeze(1)
        weights = torch.exp(-distances / temperature)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return torch.einsum("bn,bnd->bd", weights, neighbors)

    elif mode == "pca":
        B, N, D = neighbors.shape
        view2_list = []
        for b in range(B):
            X = neighbors[b]  # [N, D]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            v = Vh[0]  # top-1 direction
            proj = X @ v  # [N]
            weight = proj / (proj.sum() + 1e-8)
            pooled = (weight[:, None] * X).sum(dim=0)  # [D]
            view2_list.append(pooled)
        return torch.stack(view2_list)

    elif mode == "svd_nuclear":
        B, N, D = neighbors.shape
        view2_list = []
        for b in range(B):
            X = neighbors[b]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            nuclear_norm = S.sum()
            if nuclear_norm > 1e-6:
                weights = torch.zeros(N, device=X.device)
                for i in range(min(len(S), N)):
                    if S[i] > 1e-6:
                        proj = torch.abs(X @ Vh[i])
                        weights += (S[i] / nuclear_norm) * proj
                weights = weights / (weights.sum() + 1e-8)
            else:
                weights = torch.ones(N, device=X.device) / N
            pooled = (weights[:, None] * X).sum(dim=0)
            view2_list.append(pooled)
        return torch.stack(view2_list)

    elif mode == "cosine":
        assert anchor is not None
        sims = F.cosine_similarity(anchor.unsqueeze(1), neighbors, dim=-1)  # [B, N]
        weights = F.softmax(sims / temperature, dim=1)  # [B, N]
        return torch.einsum("bn,bnd->bd", weights, neighbors)

    else:
        raise ValueError(f"Unknown mode: {mode}")

def organize_manifolds(embeddings: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
    n_samples = embeddings.shape[0]
    n_neighbors = knn_indices.shape[1] - 1
    embedding_dim = embeddings.shape[1]
    
    assert embeddings.shape[0] == knn_indices.shape[0], "embeddings and knn_indices must have the same number of samples"

    manifolds = torch.zeros((n_samples, n_neighbors + 1, embedding_dim), device=embeddings.device)
    for i in range(n_samples):
        manifolds[i, 0] = embeddings[i]
        manifolds[i, 1:] = embeddings[knn_indices[i, 1:]]
    assert manifolds.shape == (n_samples, n_neighbors + 1, embedding_dim), "manifolds shape is not correct"
    return manifolds


def organize_manifolds_soft(embeddings, knn_indices, knn_distances, temperature: float = 0.3) -> torch.Tensor:
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if isinstance(knn_indices, np.ndarray):
        knn_indices = torch.from_numpy(knn_indices)
    if isinstance(knn_distances, np.ndarray):
        knn_distances = torch.from_numpy(knn_distances)

    n_samples = embeddings.shape[0]
    n_neighbors = knn_indices.shape[1] - 1
    embedding_dim = embeddings.shape[1]
    
    weights = F.softmax(-knn_distances[:, 1:] / temperature, dim=1)

    manifolds = torch.zeros((n_samples, n_neighbors + 1, embedding_dim), device=embeddings.device)
    neighbor_weights = torch.zeros((n_samples, n_neighbors), device=embeddings.device)
    
    for i in range(n_samples):
        manifolds[i, 0] = embeddings[i]
        manifolds[i, 1:] = embeddings[knn_indices[i, 1:]]
        neighbor_weights[i] = weights[i]

    return manifolds, neighbor_weights
    

def estimate_local_nuc(embeddings: torch.Tensor, knn_indices: np.ndarray) -> np.ndarray:
    n, d = embeddings.shape
    nuc_scores = torch.zeros(n, device=embeddings.device)

    for i in range(n):
        neighbors = embeddings[knn_indices[i]]  # [k, d]
        s = torch.linalg.svdvals(neighbors)     # [min(k, d)]
        nuc_scores[i] = s.sum()

    return nuc_scores.cpu().numpy()

