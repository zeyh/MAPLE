import numpy as np
np.infty = np.inf 
import umap
import torch
import torch.nn.functional as F
import scipy
from tqdm import tqdm
import numba
from umap.layouts import  tau_rand_int, clip

def reorginize_embeddings(ego_embeddings, neighbors_embeddings, knn_indices):
    # Given the embeddings then output a dict that for each point, all its positions in different manifolds, given the knn_indices as the reference
    n_samples = ego_embeddings.shape[0]
    point_embeddings = [[] for _ in range(n_samples)] 
    
    for manifold_idx in range(n_samples):
        ego_point = knn_indices[manifold_idx, 0]
        point_embeddings[ego_point].append(ego_embeddings[manifold_idx])
        

        neighbor_points = knn_indices[manifold_idx, 1:]
        for pos, point_idx in enumerate(neighbor_points):
            point_embeddings[point_idx].append(neighbors_embeddings[manifold_idx, pos])
    


    return point_embeddings

def compute_weighted_knn_from_reorganized(point_embeddings, n_neighbors=15):
    # Get the maximum number of charts for each point, since tensor do not support variable length, we need to pad the tensor with NaNs
    max_charts = max(len(p) for p in point_embeddings)
    n_points = len(point_embeddings)

    # if less than max_charts, pad with NaNs
    dim = point_embeddings[0][0].shape[-1]
    padded = []
    for p in point_embeddings:
        charts = p + [torch.full((dim,), float('nan'))] * (max_charts - len(p))
        padded.append(torch.stack(charts))
    positions = torch.stack(padded)  # shape: [n_points, max_charts, dim]

    # Get the mask of valid charts by blocking nans
    valid_mask = ~positions.isnan().any(dim=-1)  # [n_points, max_charts]
    dists_per_chart = []
    
    valid_chart_count = 0
    total_distance_written = 0

    for c in tqdm(range(max_charts), desc="Computing chart distances"):
        valid = valid_mask[:, c]
        if valid.sum() < 2:
            continue
        
        rows = cols = torch.where(valid)[0]
        chart_pos = positions[valid, c, :]
        chart_dist = torch.cdist(chart_pos, chart_pos)
        
        dist = torch.full((n_points, n_points), float('inf'), device=chart_pos.device)
        dist[rows[:, None], cols[None, :]] = chart_dist
        dists_per_chart.append(dist)

        valid_chart_count += 1
        total_distance_written += chart_dist.numel()

    if not dists_per_chart:
        raise ValueError("No valid charts to compute distances.")

    stacked = torch.stack(dists_per_chart)  # shape: [n_charts, n, n]
    mask = torch.isfinite(stacked)
    stacked[~mask] = float('nan')

    mean_dists = torch.nanmean(stacked, dim=0)

    valid = torch.isfinite(mean_dists)
    min_val = mean_dists[valid].min()
    max_val = mean_dists[valid].max()
    mean_dists[valid] = (mean_dists[valid] - min_val) / (max_val - min_val)

    # # use median
    # stacked[~mask] = float('nan')
    # median_dists = torch.nanmedian(stacked, dim=0).values

    # # use min
    # LARGE_DIST = 1e6
    # stacked[~mask] = LARGE_DIST  # make sure non-cooccurring distances don't dominate
    # min_dists, _ = stacked.min(dim=0)  # shape: [n_points, n_points]
    # num_valid = mask.sum(dim=0)
    # min_dists[num_valid == 0] = float('inf')

    # # use weighted mean
    # # Compute variance of pairwise distances per chart
    # variances = stacked.var(dim=(1, 2)) + 1e-8  # avoid divide-by-zero
    # weights = 1 / variances  # higher weight for more stable charts
    # weights = weights / weights.sum()  # normalize
    # weighted_mean = (stacked * weights[:, None, None]).sum(dim=0)

    new_dists = mean_dists

    # mean_dists = torch.stack(dists_per_chart).mean(dim=0)  # shape: [n_points, n_points]
    knn_dists, knn_indices = torch.topk(new_dists, k=n_neighbors, largest=False)

    return knn_indices, knn_dists


def estimate_mean_local_rank(point_embeddings, rank_threshold=0.01):
    n_points = len(point_embeddings)
    mean_ranks = torch.zeros(n_points)

    for i in range(n_points):
        charts = point_embeddings[i]
        if len(charts) < 2:
            mean_ranks[i] = 1.0 
            continue
        local_matrix = torch.stack(charts)  # [num_charts_i, D]
        svdvals = torch.linalg.svdvals(local_matrix)
        rank = (svdvals > rank_threshold).sum().item()
        mean_ranks[i] = rank
    return mean_ranks



def compute_knn_indices(embeddings, n_neighbors=15):
    distances = torch.cdist(embeddings, embeddings)
    values, indices = torch.topk(distances, k=n_neighbors, dim=1, largest=False)
    return indices, values

def build_knn_graph(X, n_neighbors=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    knn_dists, knn_indices = nbrs.kneighbors(X)
    X = torch.tensor(X, dtype=torch.float32)
    return knn_indices, knn_dists, X



def umap_original(X, n_neighbors=15, verbose=True, max_spectral_n=40001):
    """
    to avoid O(N^3) memory issues in spectral initialization.
    """
    n_samples = X.shape[0]
    
    if n_samples > max_spectral_n:
        if verbose:
            print(f"Large N detected ({n_samples} > {max_spectral_n}). Using Random Initialization.")
        init_strategy = "random"
    else:
        if verbose:
            print(f"Small N detected ({n_samples} <= {max_spectral_n}). Using Spectral Initialization.")
        init_strategy = "spectral"
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        init=init_strategy,
        low_memory=True,  
        n_jobs=-1,      
        verbose=verbose     
    )
    Z_umap = reducer.fit_transform(X)
    return Z_umap



def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    # distance = knn_dists
    SMOOTH_K_TOLERANCE = 1e-5
    MIN_K_DIST_SCALE = 1e-3
    NPY_INFINITY = np.inf
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def compute_membership_strengths(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    return_dists=False,
    bipartite=False,
):    
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def compute_relative_distances(embeddings, method="l2"):
    ego_points = embeddings[:, 0, :]
    relative_dists = torch.zeros(embeddings.shape[0], embeddings.shape[1])

    scale_factor = 1
    if method == "l2":
        for i in range(embeddings.shape[1]):
            relative_dists[:, i] = scale_factor * torch.norm(embeddings[:, i, :] - ego_points, dim=1)
    
    elif method == "l1":
        for i in range(embeddings.shape[1]):
            relative_dists[:, i] = scale_factor * torch.sum(torch.abs(embeddings[:, i, :] - ego_points), dim=1)
    
    elif method == "cosine":
        for i in range(embeddings.shape[1]):
            cos_sim = F.cosine_similarity(embeddings[:, i, :], ego_points)
            relative_dists[:, i] = scale_factor * (1 - cos_sim)
    
    return relative_dists



def flatten_knn_weights(knn_dists):
    n_samples, n_neighbors = knn_dists.shape
    vals = knn_dists.reshape(-1)  
    return vals

def reshape_back(vals):
    matrix = vals.reshape(-1, 10)
    return matrix


def softmax_knn_weights(knn_dists, temperature=1.0, local_nuc=None):
    knn_dists = knn_dists.copy()
    knn_dists[:, 0] = np.inf  # ignore self-distance

    min_dist = np.min(knn_dists, axis=1, keepdims=True)
    stable_dists = knn_dists - min_dist


    weights = np.exp(-stable_dists / temperature)
    weights[:, 0] = 0.0  # enforce self-loop = 0

    if local_nuc is not None:
        flatness = (local_nuc - np.min(local_nuc)) / (np.max(local_nuc) - np.min(local_nuc) + 1e-8)
        temperature_flatness = 1.0
        trust_weight = np.exp(-flatness / temperature_flatness) # size n
        weights *= trust_weight[:, None] # n, k

    return weights


def gen_fuzzy_simplicial_set(
    n_samples,
    n_neighbors,
    knn_indices,
    knn_dists,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    return_dists=None,
):
    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists, 
        float(n_neighbors), 
        local_connectivity=float(local_connectivity))

    rows, cols, vals, dists = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos, return_dists)

    new_vals = softmax_knn_weights(knn_dists)
    new_vals = flatten_knn_weights(new_vals)
    vals = new_vals

    result = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()
        prod_matrix = result.multiply(transpose)  # Element-wise multiplication
        result = set_op_mix_ratio * (result + transpose - prod_matrix) + (1.0 - set_op_mix_ratio) * prod_matrix
    result.eliminate_zeros()

    if return_dists is None:
        return result
    else:
        if return_dists:
            dmat = scipy.sparse.coo_matrix((dists, (rows, cols)), shape=(n_samples, n_samples))
            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result
def preprocess_fuzzy_set(graph, n_epochs=None):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    weight = graph.data
    
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    if n_epochs is None:
        n_epochs = default_epochs

    # If n_epoch is a list, get the maximum epoch to reach
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs

    if n_epochs_max > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    return graph

import umap.distances as dist
def optimize_layout_generic(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=dist.euclidean,
    output_metric_kwds=(),
    verbose=False,
    tqdm_kwds=None,
    move_other=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.

    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """


    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _optimize_layout_generic_single_epoch,
        fastmath=True,
    )

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    rng_state_per_sample = np.full(
        (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
    ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        optimize_fn(
            epochs_per_sample,
            epoch_of_next_sample,
            head,
            tail,
            head_embedding,
            tail_embedding,
            output_metric,
            output_metric_kwds,
            dim,
            alpha,
            move_other,
            n,
            epoch_of_next_negative_sample,
            epochs_per_negative_sample,
            rng_state_per_sample,
            n_vertices,
            a,
            b,
            gamma,
        )
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

    return head_embedding


def _optimize_layout_generic_single_epoch(
    epochs_per_sample,
    epoch_of_next_sample,
    head,
    tail,
    head_embedding,
    tail_embedding,
    output_metric,
    output_metric_kwds,
    dim,
    alpha,
    move_other,
    n,
    epoch_of_next_negative_sample,
    epochs_per_negative_sample,
    rng_state_per_sample,
    n_vertices,
    a,
    b,
    gamma,
):
    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_output, grad_dist_output = output_metric(
                current, other, *output_metric_kwds
            )
            _, rev_grad_dist_output = output_metric(other, current, *output_metric_kwds)

            if dist_output > 0.0:
                w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
            else:
                w_l = 1.0
            grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

            for d in range(dim):
                grad_d = clip(grad_coeff * grad_dist_output[d])

                current[d] += grad_d * alpha
                if move_other:
                    grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                    other[d] += grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state_per_sample[j]) % n_vertices

                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )

                if dist_output > 0.0:
                    w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                elif j == k:
                    continue
                else:
                    w_l = 1.0

                grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
    return epoch_of_next_sample, epoch_of_next_negative_sample


