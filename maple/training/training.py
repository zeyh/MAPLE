from numpy import ndarray
import numpy as np
import random
from tqdm import tqdm
import os
np.infty = np.inf 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from maple.models.mmcr_model import Model as MMCR_Model
from maple.training.loss_functions import MMCR_Loss, BarlowTwinsLoss
from utils.mmcr_utils import organize_manifolds, organize_manifolds_soft, build_view2


from config.default_config import DEVICE, CONFIG_SET2, CONFIG_SET0, CONFIG_SET1, CONFIG_MMCR, CONFIG_BT

def compute_knn_similarity(knn_indices_old, knn_indices_new):
    if torch.is_tensor(knn_indices_old):
        knn_indices_old = knn_indices_old.cpu().numpy()
    if torch.is_tensor(knn_indices_new):
        knn_indices_new = knn_indices_new.cpu().numpy()
    
    N = knn_indices_old.shape[0]
    jaccard_scores = []
    overlap_scores = []
    
    for i in range(N):
        old_neighbors = set(knn_indices_old[i, 1:].tolist())
        new_neighbors = set(knn_indices_new[i, 1:].tolist())
        
        intersection = len(old_neighbors.intersection(new_neighbors))
        union = len(old_neighbors.union(new_neighbors))
        
        jaccard = intersection / union if union > 0 else 0.0
        overlap = intersection / len(old_neighbors) if len(old_neighbors) > 0 else 0.0
        
        jaccard_scores.append(jaccard)
        overlap_scores.append(overlap)
    
    avg_jaccard = np.mean(jaccard_scores)
    avg_overlap = np.mean(overlap_scores)
    
    return avg_jaccard, avg_overlap

def compute_geodesic_correlation(knn_indices, z_2d, n_sample_pairs=1000, min_2d_distance=None, 
                                  max_2d_distance=None, directed=False):
    if torch.is_tensor(knn_indices):
        knn_indices = knn_indices.cpu().numpy()
    if torch.is_tensor(z_2d):
        z_2d = z_2d.cpu().numpy()
    
    N = knn_indices.shape[0]
    k = knn_indices.shape[1] - 1  # excluding self
    
    # Build sparse adjacency matrix from KNN indices
    rows = []
    cols = []
    
    for i in range(N):
        ego_idx = knn_indices[i, 0]
        neighbors = knn_indices[i, 1:]
        
        # Add edges: ego -> neighbors
        for neighbor_idx in neighbors:
            if 0 <= neighbor_idx < N:
                rows.append(ego_idx)
                cols.append(neighbor_idx)
    
    # Create sparse matrix (unweighted graph, all edges have weight 1)
    data = np.ones(len(rows), dtype=np.float64)
    graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Make symmetric if undirected
    if not directed:
        graph = graph + graph.T
        graph.data = np.ones_like(graph.data)  # Ensure all edges are weight 1
    
    # Compute shortest path distances for all pairs (or sample if too large)
    print(f"Computing shortest paths on learned KNN graph (N={N})...")
    if N > 10000:
        # For large graphs, compute shortest paths only for sampled pairs
        distances_matrix = None
        use_sampling = True
    else:
        # For smaller graphs, compute full distance matrix
        # Graph is unweighted (all edges = 1), so shortest_path computes hop count
        distances_matrix = shortest_path(
            csgraph=graph, 
            method='D', 
            directed=directed
        )
        use_sampling = False
    
    # Sample pairs of points
    max_attempts = n_sample_pairs * 10
    attempts = 0
    sampled_pairs = []
    
    while len(sampled_pairs) < n_sample_pairs and attempts < max_attempts:
        i, j = np.random.choice(N, size=2, replace=False)
        
        # Compute 2D Euclidean distance
        d_2d = np.linalg.norm(z_2d[i] - z_2d[j])
        
        # Apply distance filters if specified
        if min_2d_distance is not None and d_2d < min_2d_distance:
            attempts += 1
            continue
        if max_2d_distance is not None and d_2d > max_2d_distance:
            attempts += 1
            continue
        
        sampled_pairs.append((i, j))
        attempts += 1
    
    if len(sampled_pairs) < n_sample_pairs:
        print(f"⚠️  Only sampled {len(sampled_pairs)} pairs (requested {n_sample_pairs})")
    
    # Compute distances for sampled pairs
    graph_distances = []
    euclidean_distances = []
    
    for i, j in tqdm(sampled_pairs, desc="Computing distances"):
        d_2d = np.linalg.norm(z_2d[i] - z_2d[j])
        euclidean_distances.append(d_2d)
        
        if use_sampling:
            # Compute shortest path for this specific pair
            dist_matrix_pair = shortest_path(
                csgraph=graph,
                method='D',
                directed=directed,
                indices=[i]
            )
            d_graph = dist_matrix_pair[0, j]
        else:
            d_graph = distances_matrix[i, j]
        
        graph_distances.append(d_graph)
    
    graph_distances = np.array(graph_distances)
    euclidean_distances = np.array(euclidean_distances)
    
    # Filter out pairs with infinite or invalid graph distances
    valid_mask = np.isfinite(graph_distances) & (graph_distances > 0)
    n_valid_pairs = np.sum(valid_mask)
    
    if n_valid_pairs < 10:
        print(f"⚠️  Too few valid pairs ({n_valid_pairs}), correlation may be unreliable")
        return 0.0, graph_distances, euclidean_distances, n_valid_pairs
    
    graph_distances_valid = graph_distances[valid_mask]
    euclidean_distances_valid = euclidean_distances[valid_mask]
    
    # Compute Spearman rank correlation
    spearman_corr, p_value = spearmanr(graph_distances_valid, euclidean_distances_valid)
    
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
    
    print(f"✅ Geodesic correlation: {spearman_corr:.4f} (p={p_value:.4e}, n={n_valid_pairs} pairs)")
    
    return spearman_corr, graph_distances, euclidean_distances, n_valid_pairs

def compute_knn_gpu(z_all, k_neighbors, device, batch_size=10000):
    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False
        print("⚠️  FAISS not available, falling back to PyTorch KNN computation")
    
    N, D = z_all.shape
    k = k_neighbors + 1  # +1 for self
    
    max_safe_n_for_full_matrix = 15000
    force_batching = N > max_safe_n_for_full_matrix
    
    if use_faiss and device.type == "cuda":
        z_np = z_all.cpu().numpy().astype('float32')
        
        index = faiss.IndexFlatIP(D)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(z_np)
        
        if force_batching:
            print(f"Computing KNN with FAISS in batches (N={N}, batch_size={batch_size})...")
            all_indices = []
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                z_batch = z_np[i:end_i]
                distances, indices = index.search(z_batch, k)
                all_indices.append(indices)
            indices = np.vstack(all_indices)
        else:
            distances, indices = index.search(z_np, k)
        
        knn_indices = np.zeros((N, k), dtype=np.int64)
        for i in range(N):
            knn_indices[i, 0] = i
            neighbor_pos = 1
            for idx in indices[i]:
                if idx != i and neighbor_pos < k:
                    knn_indices[i, neighbor_pos] = idx
                    neighbor_pos += 1
    elif use_faiss:
        z_np = z_all.cpu().numpy().astype('float32')
        
        index = faiss.IndexFlatIP(D)
        index.add(z_np)
        
        if force_batching:
            print(f"Computing KNN with FAISS in batches (N={N}, batch_size={batch_size})...")
            all_indices = []
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                z_batch = z_np[i:end_i]
                distances, indices = index.search(z_batch, k)
                all_indices.append(indices)
            indices = np.vstack(all_indices)
        else:
            distances, indices = index.search(z_np, k)
        
        knn_indices = np.zeros((N, k), dtype=np.int64)
        for i in range(N):
            knn_indices[i, 0] = i
            neighbor_pos = 1
            for idx in indices[i]:
                if idx != i and neighbor_pos < k:
                    knn_indices[i, neighbor_pos] = idx
                    neighbor_pos += 1
    else:
        if force_batching or N > batch_size:
            query_batch = min(batch_size, 2000) if N > 30000 else batch_size
            ref_batch = min(batch_size, 5000)
            
            print(f"Computing KNN in batches (N={N}, query_batch={query_batch}, ref_batch={ref_batch})...")
            knn_indices = np.zeros((N, k), dtype=np.int64)
            
            for i in range(0, N, query_batch):
                end_i = min(i + query_batch, N)
                z_batch = z_all[i:end_i].to(device, non_blocking=True)
                
                topk_values = torch.full((end_i - i, k), float('-inf'), device=device)
                topk_indices = torch.zeros((end_i - i, k), dtype=torch.long, device=device)
                
                for j in range(0, N, ref_batch):
                    end_j = min(j + ref_batch, N)
                    z_ref = z_all[j:end_j].to(device, non_blocking=True)
                    sims_chunk = torch.mm(z_batch, z_ref.t())
                    
                    batch_size_actual = end_i - i
                    for idx in range(batch_size_actual):
                        if i + idx >= j and i + idx < end_j:
                            sims_chunk[idx, i + idx - j] = -float('inf')
                    
                    ref_indices = torch.arange(j, end_j, dtype=torch.long, device=device)
                    
                    chunk_topk_values, chunk_topk_pos = torch.topk(sims_chunk, k=min(k, end_j - j), dim=1, largest=True)
                    chunk_topk_indices = ref_indices[chunk_topk_pos]
                    
                    combined_values = torch.cat([topk_values, chunk_topk_values], dim=1)
                    combined_indices = torch.cat([topk_indices, chunk_topk_indices], dim=1)
                    
                    topk_values, topk_pos = torch.topk(combined_values, k=k, dim=1, largest=True)
                    topk_indices = torch.gather(combined_indices, 1, topk_pos)
                
                knn_indices[i:end_i, 0] = np.arange(i, end_i)
                knn_indices[i:end_i, 1:] = topk_indices.cpu().numpy()[:, :k-1]
        else:
            z_all_device = z_all.to(device)
            sims = torch.mm(z_all_device, z_all_device.t())
            sims.fill_diagonal_(-float('inf'))
            _, indices = torch.topk(sims, k=k, dim=1, largest=True)
            knn_indices = indices.cpu().numpy()
            knn_indices[:, 0] = np.arange(N)
    
    return knn_indices

def training_with_bt_dynamic_knn(raw_X, raw_knn_indices, ):
    # * TRraining parameters
    batch_size = CONFIG_SET0["batch_size"]
    num_workers = 4
    lr = CONFIG_SET0["learning_rate"]
    weight_decay = CONFIG_SET0["weight_decay"]    
    KNN_Embedding_dim = CONFIG_SET0["KNN_Embedding_dim"]

    epochs = CONFIG_SET1["epochs"]
    update_knn_every = CONFIG_SET2["update_knn_every"]  # New parameter
    n_neighbors = raw_knn_indices.shape[1] - 1  # Excluding ego

    # * Data normalization
    raw_X: Tensor = torch.nn.functional.normalize(raw_X, p=2, dim=1)
    
    # Initial KNN setup
    current_knn_indices = raw_knn_indices
    manifolds = organize_manifolds(raw_X, current_knn_indices)
    
    manifold_size_gb = (manifolds.numel() * 4) / (1024**3)
    use_pin_memory = CONFIG_SET2.get("use_pin_memory", True)
    use_persistent_workers = CONFIG_SET2.get("use_persistent_workers", True)
    
    if manifold_size_gb > 8.0:
        print(f"⚠️  Large tensor detected ({manifold_size_gb:.2f} GB)!!")
        use_pin_memory = False
        use_persistent_workers = False
    
    train_dataset = TensorDataset(manifolds)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers if use_pin_memory else False
    )

    # model = MMCR_Model(input_dim=raw_X.shape[1], projector_dims=[KNN_Embedding_dim, raw_X.shape[1]])
    
    if False: 
        # stl10
        # model = MMCR_Model(input_dim=raw_X.shape[1], use_resnet=True, dataset="stl10", image_shape=(3, 96, 96))
        # cifar 10
        # model = MMCR_Model(input_dim=raw_X.shape[1], use_resnet=True, dataset="cifar10", image_shape=(3, 32, 32))
        # mnist digits
        model = MMCR_Model(input_dim=raw_X.shape[1], use_resnet=True, dataset="conv1", image_shape=(1, 28, 28))
        # fashion mnist 
        # model = MMCR_Model(input_dim=raw_X.shape[1], use_resnet=True, dataset="conv1", image_shape=(1, 28, 28))

    else:
        model = MMCR_Model(input_dim=raw_X.shape[1])



    device = DEVICE
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bt_loss = BarlowTwinsLoss(device=device, lambda_param=0.005)
    mmcr_loss = MMCR_Loss(lmbda=CONFIG_MMCR["lmbda"], n_aug=n_neighbors+1, distributed=False)
    
    device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    use_amp = CONFIG_SET2.get("use_amp", True) if device_type == "cuda" else False
    scaler = GradScaler(enabled=use_amp)

    # * Training ==============================================
    latest_embedding_ego = None
    latest_embedding_neighbors = None
    
    all_mu_values = []
    all_knn_indices_batch = []
    
    knn_stability_history = []
    previous_knn_indices = None
    
    use_dimension_shift = CONFIG_SET2.get("use_dimension_shift", False)
    shift_warmup_epochs = CONFIG_SET2.get("shift_warmup_epochs", 2)
    has_shifted_to_z = False
    shift_epoch = None
    current_input_data = raw_X

    print(f"start training -----")
    if use_dimension_shift:
        print(f"🚀 shift from raw_X to z_all after first KNN rebuild")
    
    for epoch in range(epochs):
        if update_knn_every != -1 and epoch % update_knn_every == 0 and epoch > 0:
            print(f"🔁 rebuilding KNN at epoch {epoch}")
            model.eval()
            with torch.no_grad():
                z_all = []
                for i in range(0, current_input_data.size(0), batch_size):
                    xb = current_input_data[i:i+batch_size].to(device, non_blocking=True)
                    _, z = model(xb)
                    z = F.normalize(z, dim=1)
                    z_all.append(z.cpu())
                z_all = torch.cat(z_all, dim=0)
            
            if CONFIG_SET2["manual_recompute_knn_on_Z"]: # and epoch > 11:  #! NOT USED! 
                current_knn_indices, current_mu_values = compute_learned_neighborhoods_and_weights(
                    z_all, k_neighbors=n_neighbors, temperature=0.1, device=device
                )
            else:
                use_gpu_knn = CONFIG_SET2.get("use_gpu_knn", True)
                if use_gpu_knn and device.type in ["mps", "cuda"]:
                    current_knn_indices = compute_knn_gpu(
                        z_all, k_neighbors=n_neighbors, device=device, batch_size=10000
                    )
                else:
                    knn = NearestNeighbors(
                        n_neighbors=n_neighbors + 1, 
                        metric="cosine",
                        n_jobs=-1
                    )
                    knn.fit(z_all.numpy())
                    current_knn_indices = knn.kneighbors(z_all.numpy(), return_distance=False)

            stability_info = {'epoch': epoch}
            if previous_knn_indices is not None:
                jaccard_sim, overlap_ratio = compute_knn_similarity(previous_knn_indices, current_knn_indices)
                print(f"📊 Graph vs Previous: Jaccard={jaccard_sim:.2%}, Overlap={overlap_ratio:.2%}")
                stability_info['jaccard_vs_previous'] = jaccard_sim
                stability_info['overlap_vs_previous'] = overlap_ratio
            jaccard_vs_initial, overlap_vs_initial = compute_knn_similarity(raw_knn_indices, current_knn_indices)
            print(f"📊 Graph vs Initial: Jaccard={jaccard_vs_initial:.2%}, Overlap={overlap_vs_initial:.2%}")
            stability_info['jaccard_vs_initial'] = jaccard_vs_initial
            stability_info['overlap_vs_initial'] = overlap_vs_initial
            
            knn_stability_history.append(stability_info)
            
            if isinstance(current_knn_indices, np.ndarray):
                previous_knn_indices = current_knn_indices.copy()
            else:
                previous_knn_indices = current_knn_indices.clone()

            # Dimension shift logic
            if use_dimension_shift and not has_shifted_to_z:
                print(f"🚀 Input ({current_input_data.shape[1]}D) -> Latent ({z_all.shape[1]}D)")
                
                old_state_dict = model.state_dict()
                new_model = MMCR_Model(input_dim=z_all.shape[1]).to(device)
                
                new_state_dict = new_model.state_dict()
                for name, param in old_state_dict.items():
                    if name in new_state_dict and param.size() == new_state_dict[name].size():
                        new_state_dict[name].copy_(param)
                        print(f"✅ Preserved layer: {name}")
                
                model = new_model
                shift_epoch = epoch
                has_shifted_to_z = True
                current_input_data = z_all.detach()
                
                warmup_lr = lr * 0.1
                optimizer = torch.optim.Adam(model.parameters(), lr=warmup_lr, weight_decay=weight_decay)
                print(f"✅ Shift complete. Using warmup LR={warmup_lr:.6f} for {shift_warmup_epochs} epochs")
            elif use_dimension_shift and has_shifted_to_z:
                current_input_data = z_all.detach()
                
                if epoch - shift_epoch < shift_warmup_epochs:
                    warmup_lr = lr * 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                elif epoch - shift_epoch == shift_warmup_epochs:
                    post_warmup_lr = lr * 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = post_warmup_lr
                    print(f"✅ Warmup complete. Switching to post-shift LR={post_warmup_lr:.6f}")
                else:
                    print(f"✅ Refined Topology on Latent Space. Weights preserved.")

            # Re-organize manifolds using current input data
            if use_dimension_shift and has_shifted_to_z:
                manifolds = organize_manifolds(current_input_data, current_knn_indices)
                print(f"✅ Topology updated on latent space.")
            else:
                manifolds = organize_manifolds(raw_X, current_knn_indices)
                print(f"✅ Topology updated. Continuing training with original weights.")


            manifold_size_gb = (manifolds.numel() * 4) / (1024**3)
            use_pin_memory = CONFIG_SET2.get("use_pin_memory", True)
            use_persistent_workers = CONFIG_SET2.get("use_persistent_workers", True)
            
            if manifold_size_gb > 8.0:
                use_pin_memory = False
                use_persistent_workers = False
            
            train_dataset = TensorDataset(manifolds)
            train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                persistent_workers=use_persistent_workers if use_pin_memory else False
            )


        model.train()
        total_loss, total_num = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        is_last_epoch = (epoch == epochs - 1)
        if is_last_epoch:
            num_samples = current_knn_indices.shape[0]
            epoch_embeddings_ego = None
            epoch_embeddings_neighbors = None
            epoch_mu_values = None
            epoch_knn_indices = None
            feature_dim = None
            num_neighbors = None

        for step, batch_data in enumerate(train_bar):
            optimizer.zero_grad()

            if len(batch_data) == 2:
                manifold_embeddings, batch_weights = batch_data
                batch_weights = batch_weights.to(device, non_blocking=True)
            else:
                manifold_embeddings = batch_data[0]
                batch_weights = None

            manifold_embeddings = manifold_embeddings.float().to(device, non_blocking=True)
            # manifold_embeddings = manifold_embeddings.to(device, non_blocking=True)

            B, N, D = manifold_embeddings.shape
            
            with autocast(device_type=device_type, enabled=use_amp):
                manifold_embeddings = manifold_embeddings.view(B * N, D)
                features, out = model(manifold_embeddings)
                #print(f"features.shape: {features.shape}")
                #print(f"out.shape: {out.shape}")
                out_reshaped = out.view(B, N, -1)
                view1 = out_reshaped[:, 0, :]  # anchor
                neighbors = out_reshaped[:, 1:, :]

                # * Compute learned mu values with threshold
                if CONFIG_SET2["adaptive_mu_temperature"]:
                    temperature = get_temperature(epoch, max_epochs=epochs)
                else: 
                    temperature = 0.1
                mu_values = compute_learned_mu_values(view1, neighbors, temperature=temperature, threshold=0.5)

                # * Use learned mu values to construct view2 
                if CONFIG_BT["use_learned_mu_for_view2"]: 
                    view2 = torch.einsum("bn,bnd->bd", mu_values, neighbors)  # [B, D]
                else: 
                    view2 = build_view2(neighbors=neighbors, mode=CONFIG_BT["view2_mode"], anchor=view1)

                if CONFIG_SET2["use_bt_loss"] == "True": 
                    bt_loss_value = bt_loss(view1, view2) 
                    loss = 1 * bt_loss_value
                    mmcr_loss_dict = None
                elif CONFIG_SET2["use_bt_loss"] == "False":
                    mmcr_loss_value, mmcr_loss_dict = mmcr_loss(out, batch_weights)
                    loss = 1 * mmcr_loss_value # + 0.001 * mmcr_loss_value 
                elif CONFIG_SET2["use_bt_loss"] == "Both":
                    bt_loss_value = bt_loss(view1, view2)
                    mmcr_loss_value, mmcr_loss_dict = mmcr_loss(out, batch_weights)
                    loss = CONFIG_SET2["bt_loss_weight"] * bt_loss_value + CONFIG_SET2["mmcr_loss_weight"] * mmcr_loss_value
                else:
                    raise ValueError("Invalid use_bt_loss value!")
            
            # * Initialize pre-allocated tensors on first batch of last epoch
            if is_last_epoch and step == 0:
                feature_dim = features.shape[-1]
                num_neighbors = neighbors.shape[1]
                epoch_embeddings_ego = torch.zeros(num_samples, feature_dim, dtype=features.dtype)
                epoch_embeddings_neighbors = torch.zeros(num_samples, num_neighbors, feature_dim, dtype=features.dtype)
                epoch_mu_values = torch.zeros(num_samples, num_neighbors, dtype=mu_values.dtype)
                epoch_knn_indices = torch.zeros(num_samples, current_knn_indices.shape[1], dtype=torch.long)
            
            # * Store mu and current knn_indices for this batch (outside autocast)
            if is_last_epoch:
                batch_start = step * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                mu_values_cpu = mu_values.detach().cpu()
                epoch_mu_values[batch_start:batch_end] = mu_values_cpu
                
                batch_knn_indices = current_knn_indices[batch_start:batch_end]
                if isinstance(batch_knn_indices, np.ndarray):
                    epoch_knn_indices[batch_start:batch_end] = torch.from_numpy(batch_knn_indices)
                else:
                    epoch_knn_indices[batch_start:batch_end] = batch_knn_indices.cpu()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_num += manifold_embeddings.size(0)
            total_loss += loss.item() * manifold_embeddings.size(0)
            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.4f}".format(
                    epoch, epochs, total_loss / total_num,
                )
            )
            
            # * Store embeddings for last epoch only
            if is_last_epoch:
                features = features.view(B, N, -1)
                batch_start = step * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                features_cpu = features.detach().cpu()
                epoch_embeddings_ego[batch_start:batch_end] = features_cpu[:, 0]
                epoch_embeddings_neighbors[batch_start:batch_end] = features_cpu[:, 1:]

        # * Store embeddings and mu for final epoch only
        if is_last_epoch:
            latest_embedding_ego = epoch_embeddings_ego
            latest_embedding_neighbors = epoch_embeddings_neighbors
            all_mu_values = [epoch_mu_values]
            all_knn_indices_batch = [epoch_knn_indices]

    # * Compute final KNN and compare with initial vanilla KNN
    print(f"📊 Final KNN stability...")
    model.eval()
    with torch.no_grad():
        z_final = []
        for i in range(0, current_input_data.size(0), batch_size):
            xb = current_input_data[i:i+batch_size].to(device, non_blocking=True)
            _, z = model(xb)
            z = F.normalize(z, dim=1)
            z_final.append(z.cpu())
        z_final = torch.cat(z_final, dim=0)
    
    use_gpu_knn = CONFIG_SET2.get("use_gpu_knn", True)
    if use_gpu_knn and device.type in ["mps", "cuda"]:
        final_knn_indices = compute_knn_gpu(
            z_final, k_neighbors=n_neighbors, device=device, batch_size=10000
        )
    else:
        knn = NearestNeighbors(
            n_neighbors=n_neighbors + 1, 
            metric="cosine",
            n_jobs=-1
        )
        knn.fit(z_final.numpy())
        final_knn_indices = knn.kneighbors(z_final.numpy(), return_distance=False)
    
    jaccard_final, overlap_final = compute_knn_similarity(raw_knn_indices, final_knn_indices)
    print(f"📊 Final Graph vs Initial: Jaccard={jaccard_final:.2%}, Overlap={overlap_final:.2%}")
    
    final_stability_info = {
        'epoch': 'final',
        'jaccard_vs_initial': jaccard_final,
        'overlap_vs_initial': overlap_final
    }
    
    if previous_knn_indices is not None:
        jaccard_vs_last, overlap_vs_last = compute_knn_similarity(previous_knn_indices, final_knn_indices)
        print(f"📊 Final Graph vs Last Update: Jaccard={jaccard_vs_last:.2%}, Overlap={overlap_vs_last:.2%}")
        final_stability_info['jaccard_vs_last'] = jaccard_vs_last
        final_stability_info['overlap_vs_last'] = overlap_vs_last
    
    knn_stability_history.append(final_stability_info)

    # * Build fuzzy set from learned mu values
    fuzzy_set = build_fuzzy_set_from_mu_batches(all_mu_values, all_knn_indices_batch, raw_X.shape[0])
    
    # * Compute PR and R post-training
    if False: #raw_X.shape[0] < 4001:
        pr_array, r_array = compute_pr_and_r_from_fuzzy_set(latest_embedding_ego, fuzzy_set, device=device)
    else:
        pr_array, r_array = None, None
    
    print("🍉 knn stability history: ", knn_stability_history)
    return latest_embedding_ego, latest_embedding_neighbors, fuzzy_set, pr_array, r_array, final_knn_indices


def compute_pr_and_r_from_fuzzy_set(embeddings, fuzzy_set, min_neighbors=2, batch_size=100, device=None):
    if torch.is_tensor(embeddings):
        embeddings_tensor = embeddings
        if device is None:
            device = embeddings_tensor.device
    else:
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        if device is None:
            device = torch.device("cpu")
    
    embeddings_tensor = embeddings_tensor.to(device)
    n_points, C = embeddings_tensor.shape
    pr_array = torch.zeros(n_points, device=device)
    r_array = torch.zeros(n_points, device=device)
    
    if hasattr(fuzzy_set, 'tocsr'):
        fuzzy_csr = fuzzy_set.tocsr()
    else:
        fuzzy_csr = fuzzy_set
    
    print(f"Computing PR and R for {n_points} points...")
    
    for batch_start in tqdm(range(0, n_points, batch_size), desc="Computing PR and R"):
        batch_end = min(batch_start + batch_size, n_points)
        batch_indices = list(range(batch_start, batch_end))
        
        batch_covs = []
        valid_mask = []
        
        for point_idx in batch_indices:
            row = fuzzy_csr.getrow(point_idx)
            neighbor_indices = row.indices
            
            if len(neighbor_indices) >= min_neighbors:
                point_embedding = embeddings_tensor[point_idx:point_idx+1, :]
                neighbor_embeddings = embeddings_tensor[neighbor_indices, :]
                local_manifold = torch.cat([point_embedding, neighbor_embeddings], dim=0)
                N_local = local_manifold.shape[0]
                
                local_mean = local_manifold.mean(dim=0, keepdim=True)
                local_centered = local_manifold - local_mean
                
                if N_local > 1:
                    cov = (local_centered.T @ local_centered) / (N_local - 1)
                else:
                    cov = torch.zeros(C, C, device=device)
                
                batch_covs.append(cov)
                valid_mask.append(True)
            else:
                batch_covs.append(torch.zeros(C, C, device=device))
                valid_mask.append(False)
        
        if len(batch_covs) > 0:
            batch_covs_tensor = torch.stack(batch_covs)
            original_dtype = batch_covs_tensor.dtype
            
            try:
                eigenvals_batch = torch.linalg.eigvalsh(batch_covs_tensor.float())
            except (RuntimeError, NotImplementedError):
                eigenvals_batch = torch.linalg.eigvalsh(batch_covs_tensor.cpu().float())
            
            eigenvals_batch = eigenvals_batch.to(dtype=original_dtype, device=device)
            eigenvals_batch = torch.clamp(eigenvals_batch, min=0.0)
            
            l1_norms = eigenvals_batch.sum(dim=1)
            l2_norms_sq = (eigenvals_batch ** 2).sum(dim=1)
            
            pr_values = torch.where(
                l2_norms_sq > 1e-10,
                (l1_norms ** 2) / l2_norms_sq,
                torch.zeros_like(l1_norms)
            )
            r_values = torch.sqrt(l2_norms_sq)
            
            for i, point_idx in enumerate(batch_indices):
                if valid_mask[i]:
                    pr_array[point_idx] = pr_values[i]
                    r_array[point_idx] = r_values[i]
                else:
                    pr_array[point_idx] = 0.0
                    r_array[point_idx] = 0.0
    
    print(f"✅ PR and R computation complete")
    return pr_array.cpu(), r_array.cpu()


def get_temperature(epoch, max_epochs=100, T_init=0.2, T_final=0.05):
    ratio = min(epoch / max_epochs, 1.0)
    return T_init * (1 - ratio) + T_final * ratio

def compute_learned_mu_values(view1, neighbors, temperature=0.05, threshold=0.5, min_k=2):
    # # Compute cosine similarities between ego and each neighbor
    # sims = F.cosine_similarity(view1.unsqueeze(1), neighbors, dim=-1)  # [B, N-1]
    
    # # Always keep top-k neighbors regardless of threshold
    # topk_mask = torch.zeros_like(sims, dtype=torch.bool)
    # topk_indices = sims.topk(k=min(min_k, sims.size(1)), dim=1).indices  # [B, min_k]
    # topk_mask.scatter_(1, topk_indices, True)
    
    # # Apply threshold mask , only keep neighbors with similarity > threshold
    # threshold_mask = sims > threshold
    
    # # keep neighbors that are either top-k OR above threshold
    # mask = topk_mask | threshold_mask
    # sims[~mask] = -float("inf")  # Zero weight in softmax

    # # ! Compute cosine similarities between ego and each neighbor
    sims = F.cosine_similarity(view1.unsqueeze(1), neighbors, dim=-1)  # [B, N-1]
    sims = sims + 1e-6
    mu_values = F.softmax(sims / temperature, dim=1)  # [B, N-1]
    return mu_values

def build_fuzzy_set_from_mu_batches(mu_batches, knn_batches, n_points, symmetrize=True):
    from scipy.sparse import coo_matrix
    
    mu_all = torch.cat(mu_batches, dim=0).numpy()
    knn_all = torch.cat(knn_batches, dim=0).numpy()
    
    N, K = mu_all.shape
    
    rows = np.repeat(knn_all[:, 0], K)
    cols = knn_all[:, 1:].flatten()
    data = mu_all.flatten()
    
    fuzzy = coo_matrix((data, (rows, cols)), shape=(n_points, n_points))

    if symmetrize:
        fuzzy = fuzzy.tocsr()
        fuzzy = fuzzy + fuzzy.T - fuzzy.multiply(fuzzy.T)
        fuzzy = fuzzy.tocoo()
        
    return fuzzy

def compute_learned_neighborhoods_and_weights(z_all, k_neighbors=15, temperature=0.1, device=None, batch_size=5000):
    if device is None:
        device = z_all.device if torch.is_tensor(z_all) else torch.device("cpu")
    
    if torch.is_tensor(z_all):
        z_all_device = z_all.to(device)
    else:
        z_all_device = torch.tensor(z_all, device=device)
    
    N, D = z_all_device.shape
    
    if N <= batch_size:
        sims = F.cosine_similarity(z_all_device.unsqueeze(1), z_all_device.unsqueeze(0), dim=-1)
        sims.fill_diagonal_(-float("inf"))
        mu_full = F.softmax(sims / temperature, dim=1)
        
        topk_values, topk_indices = mu_full.topk(k=k_neighbors, dim=1)
        mu_values = torch.gather(mu_full, 1, topk_indices)
        
        knn_indices = torch.zeros(N, k_neighbors + 1, dtype=torch.long, device=device)
        for i in range(N):
            knn_indices[i, 0] = i
            knn_indices[i, 1:] = topk_indices[i]
        
        return knn_indices.cpu(), mu_values.cpu()
    else:
        print(f"Computing learned neighborhoods in batches (N={N}, batch_size={batch_size})...")
        all_mu_values = []
        all_knn_indices = []
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            z_batch = z_all_device[i:end_i]
            
            sims_batch = F.cosine_similarity(z_batch.unsqueeze(1), z_all_device.unsqueeze(0), dim=-1)
            batch_size_actual = end_i - i
            for j in range(batch_size_actual):
                sims_batch[j, i + j] = -float("inf")
            mu_full_batch = F.softmax(sims_batch / temperature, dim=1)
            
            topk_values_batch, topk_indices_batch = mu_full_batch.topk(k=k_neighbors, dim=1)
            mu_values_batch = torch.gather(mu_full_batch, 1, topk_indices_batch)
            
            knn_indices_batch = torch.zeros(end_i - i, k_neighbors + 1, dtype=torch.long, device=device)
            for j in range(end_i - i):
                knn_indices_batch[j, 0] = i + j
                knn_indices_batch[j, 1:] = topk_indices_batch[j]
            
            all_mu_values.append(mu_values_batch.cpu())
            all_knn_indices.append(knn_indices_batch.cpu())
        
        knn_indices = torch.cat(all_knn_indices, dim=0)
        mu_values = torch.cat(all_mu_values, dim=0)
        
        return knn_indices, mu_values



