import torch
import numpy as np


experiments_list = ["STL_10_13000_15_20250827_153823", 
                    "mnist_test_10000_15_20250826_152503", 
                    "fashion_mnist_full_40000_15_20250828_010753",
                    "celegans_l2_15000_30_20250829_002309", 
                    "celegans_l2_neuron_12029_20_20250901_143716",
                    "celegans_l2_neuron_12029_20_20250831_013202",
                    "celegans_l2_glia_3259_20_20250831_010032",
                    "celegans_processed_6188_20_20250902_134946",
                    "scp2745_4000_20_20250831_172917",
                    ]


def load_results_replot(data_id=3):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from utils.data_io import load_experiment_for_evaluation
    from vis.embedding_plot import vis_2_embedings

    experiment_id = experiments_list[data_id]
    
    
    results_dir = f"results/{experiment_id}"
    results = load_experiment_for_evaluation(experiment_id, base_save_dir="results")
    Z_umap = results["Z_umap"]
    Z_umap_original = results["Z_umap_original"]
    raw_X = results["raw_X"]
    y = results["y"]

    return Z_umap, Z_umap_original, raw_X, y


def knn_purity(knn_indices, labels):
    if not torch.is_tensor(knn_indices):
        knn_indices = torch.as_tensor(knn_indices)
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    n, k = knn_indices.shape
    purity = []
    for i in range(n):
        neighbor_labels = labels[knn_indices[i]]
        same_label_count = (neighbor_labels ==  labels[i]).float().sum()
        purity.append(same_label_count / k)
    return torch.tensor(purity).mean().item()


def compute_neighborhood_purity(Z_umap, y, n_neighbors=15):
    import torch
    
    if not torch.is_tensor(Z_umap):
        Z_umap = torch.from_numpy(Z_umap)
    if not torch.is_tensor(y):
        y = torch.from_numpy(y)

    dist_matrix = torch.cdist(Z_umap, Z_umap)
    _, knn_indices = torch.topk(dist_matrix, k=n_neighbors, dim=1, largest=False)
    knn_indices = knn_indices[:, 1:]  # [n_samples, n_neighbors]
    average_purity = knn_purity(knn_indices, y)
    return average_purity



def compute_distance_consistency(emb, label):
    if torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()
        

    point_num = emb.shape[0]
    label_num = np.unique(label).shape[0]

    centroids = np.zeros((label_num, emb.shape[1]))
    for i in range(label_num):
        centroids[i] = np.mean(emb[label == i], axis=0)
        
    consistent_num = 0
    for idx in range(point_num):
        current_label = -1
        current_dist = 1e10
        for c_idx in range(len(centroids)):
            dist = np.linalg.norm(emb[idx] - centroids[c_idx])
            if dist < current_dist:
                current_dist = dist
                current_label = c_idx
        if current_label == label[idx]:
            consistent_num += 1
        
    return consistent_num / point_num



def validation_measure(emb, label, measure="silhouette"):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    if measure == "silhouette":
        score = silhouette_score(emb, label)
    elif measure == "calinski_harabasz": # higher the better
        score = calinski_harabasz_score(emb, label)
    elif measure == "davies_bouldin": # lower the better
        score = davies_bouldin_score(emb, label)
    return score


def clustering_based_metrics(emb, label, measure="arand",  clustering="kmeans", clustering_args=None):
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score

    if clustering_args is None:
        clustering_args = {}
    if clustering == "kmeans":
        clustering_result = KMeans(**clustering_args).fit(emb)
            
    elif clustering == "dbscan":
        clustering_result = DBSCAN(**clustering_args).fit(emb)
    else:
        raise ValueError("Invalid clustering algorithm")

    if measure == "arand":
        score = adjusted_rand_score(label, clustering_result.labels_)
    elif measure == "ami":
        score = adjusted_mutual_info_score(label, clustering_result.labels_)
    elif measure == "nmi":
        score = normalized_mutual_info_score(label, clustering_result.labels_)
    elif measure == "vmeasure":
        score = v_measure_score(label, clustering_result.labels_)
            
    return score



def local_label_trustworthiness(emb, y, k=20, return_local=False, boundary_alpha=None):
    import torch
    
    # Ensure inputs are torch tensors
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)
    
    n = emb.shape[0]
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(emb, emb)
    # Get k+1 nearest neighbors (including self)
    _, knn_indices = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
    idx = knn_indices[:, 1:]  # drop self, shape (n, k)
    
    # ranks are just positions 1..k in the neighbor list
    ranks = torch.arange(1, k+1, dtype=torch.float32, device=emb.device)[None, :]  # (1, k)
    # impostor mask
    impostor = (y[idx] != y[:, None])  # (n, k), True if different label
    
    # intrusion penalty per point: sum_{p} 1[impostor]*(k+1 - p)
    penalty_weights = (k + 1 - ranks)  # (1, k)
    P = (impostor.float() * penalty_weights).sum(axis=1)  # (n,)
    
    # per-point LT in [0,1]
    denom = k * (k + 1) / 2.0
    lt_local = 1.0 - (P / denom) * 2.0  # = 1 - 2*P/(k(k+1))
    
    # boundary filtering (optional)
    if boundary_alpha is not None:
        same_frac = (y[idx] == y[:, None]).float().mean(axis=1)
        boundary_mask = (same_frac >= boundary_alpha) & (same_frac <= 1 - boundary_alpha)
        if boundary_mask.any():
            lt = lt_local[boundary_mask].mean()
        else:
            lt = torch.tensor(1.0, device=emb.device)
    else:
        lt = lt_local.mean()
    
    # if return_local:
    #     out = {"label_trustworthiness": float(lt), "local_label_trustworthiness": lt_local.cpu().numpy()}
    # else:
    #     out = {"label_trustworthiness": float(lt)}
    return float(lt)


def hausdorff_quantile(A, B, q=0.95):
    dist = torch.cdist(A, B)
    d_ab = dist.min(dim=1).values
    d_ba = dist.min(dim=0).values
    all_d = torch.cat([d_ab, d_ba])
    return torch.quantile(all_d, q).item()

def hausdorff_metric(emb, y, q=0.95):
    if not torch.is_tensor(emb): emb = torch.as_tensor(emb, dtype=torch.float32)
    if not torch.is_tensor(y):   y   = torch.as_tensor(y)
    classes = torch.unique(y)
    results, dists = {}, []
    for i, ci in enumerate(classes):
        for cj in classes[i+1:]:
            A, B = emb[y==ci], emb[y==cj]
            hdq = hausdorff_quantile(A, B, q=q)
            results[(int(ci), int(cj))] = hdq
            dists.append(hdq)
    dists = torch.tensor(dists)
    return float(dists.median())


def run_metrics_section_simple(data_id=3):
    Z_umap, Z_umap_original, raw_X, y = load_results_replot(data_id)

    import torch
    
    if not torch.is_tensor(Z_umap):
        Z_umap = torch.from_numpy(Z_umap)
    if not torch.is_tensor(y):
        y = torch.from_numpy(y)

    neighborhood_purity = compute_neighborhood_purity(Z_umap, y, n_neighbors=15)
    print(f"MAPLE Neighborhood Purity: {neighborhood_purity}")

    neighborhood_purity_original = compute_neighborhood_purity(Z_umap_original, y, n_neighbors=15)
    print(f"Original Neighborhood Purity: {neighborhood_purity_original}")

    distance_consistency = compute_distance_consistency(Z_umap, y)
    print(f"MAPLE Distance Consistency: {distance_consistency}")

    distance_consistency_original = compute_distance_consistency(Z_umap_original, y)
    print(f"Original Distance Consistency: {distance_consistency_original}")

    silhouette = validation_measure(Z_umap, y, measure="silhouette")
    print(f"MAPLE Silhouette: {silhouette}")
    silhouette_original = validation_measure(Z_umap_original, y, measure="silhouette")
    print(f"Original Silhouette: {silhouette_original}")

    local_label_trustworthiness_val = local_label_trustworthiness(Z_umap, y, k=20, return_local=True)
    print(f"MAPLE Local Label Trustworthiness: {local_label_trustworthiness_val}")
    local_label_trustworthiness_original_val = local_label_trustworthiness(Z_umap_original, y, k=20, return_local=True)
    print(f"Original Local Label Trustworthiness: {local_label_trustworthiness_original_val}")


    calinski_harabasz = validation_measure(Z_umap, y, measure="calinski_harabasz")
    print(f"MAPLE Calinski Harabasz: {calinski_harabasz}")
    calinski_harabasz_original = validation_measure(Z_umap_original, y, measure="calinski_harabasz")
    print(f"Original Calinski Harabasz: {calinski_harabasz_original}")

    davies_bouldin = validation_measure(Z_umap, y, measure="davies_bouldin")
    print(f"MAPLE Davies Bouldin: {davies_bouldin}")
    davies_bouldin_original = validation_measure(Z_umap_original, y, measure="davies_bouldin")
    print(f"Original Davies Bouldin: {davies_bouldin_original}")

    hausdorff_distance_val = hausdorff_metric(Z_umap, y)
    print(f"MAPLE Hausdorff Distance: {hausdorff_distance_val}")
    hausdorff_distance_original = hausdorff_metric(Z_umap_original, y)
    print(f"Original Hausdorff Distance: {hausdorff_distance_original}")


    clustering_methods = ["kmeans", "dbscan"]
    clustering_metrics = ["arand", "ami", "nmi", "vmeasure"]

    for clustering_method in clustering_methods:
        for clustering_metric in clustering_metrics:
            if clustering_method == "kmeans":
                clustering_args = {"n_clusters": 10}
            elif clustering_method == "dbscan":
                clustering_args = {"eps": 0.5, "min_samples": 5}
            
            score = clustering_based_metrics(Z_umap, y, measure=clustering_metric, clustering=clustering_method, clustering_args=clustering_args)
            print(f"MAPLE {clustering_method} {clustering_metric}: {score}")
            score_original = clustering_based_metrics(Z_umap_original, y, measure=clustering_metric, clustering=clustering_method, clustering_args=clustering_args)
            print(f"Original {clustering_method} {clustering_metric}: {score_original}")



def afc_accuracy(Z, y, n_trials=20000, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    n = Z.shape[0]
    y = np.asarray(y)
    acc = 0; tot = 0
    for _ in range(n_trials):
        i = rng.integers(n)
        same = np.where(y == y[i])[0]
        diff = np.where(y != y[i])[0]
        if len(same) <= 1 or len(diff) == 0: 
            continue
        j = rng.choice(same[same != i])
        k = rng.choice(diff)
        dij = np.linalg.norm(Z[i]-Z[j])
        dik = np.linalg.norm(Z[i]-Z[k])
        acc += (dij < dik) + 0.5*(dij == dik)
        tot += 1
    return acc / max(tot,1)


def pairwise_auc_and_dprime(Z, y, max_pairs=200000, rng=None):
    from sklearn.metrics import roc_auc_score

    if rng is None: rng = np.random.default_rng(0)
    n = Z.shape[0]
    y = np.asarray(y)
    I = rng.integers(0, n, size=max_pairs)
    J = rng.integers(0, n, size=max_pairs)
    mask = I < J
    I, J = I[mask], J[mask]
    d = np.linalg.norm(Z[I]-Z[J], axis=1)
    s = (y[I] == y[J]).astype(int)

    auc = roc_auc_score(s, -d) 
    d_pos = d[s==1]; d_neg = d[s==0]
    mu_pos, mu_neg = d_pos.mean(), d_neg.mean()
    sd_pos, sd_neg = d_pos.std(ddof=1), d_neg.std(ddof=1)
    dprime = (mu_neg - mu_pos) / np.sqrt(0.5*(sd_pos**2 + sd_neg**2))
    return auc, dprime



def perceptual_similarity_metrics(Z_umap, Z_umap_original, y):
    afc_maple = afc_accuracy(Z_umap, y)
    afc_original = afc_accuracy(Z_umap_original, y)
    
    auc_maple, dprime_maple = pairwise_auc_and_dprime(Z_umap, y)
    auc_original, dprime_original = pairwise_auc_and_dprime(Z_umap_original, y)
    
    print(f"MAPLE - 2AFC: {afc_maple:.3f}, AUC: {auc_maple:.3f}, d': {dprime_maple:.3f}")
    print(f"Original - 2AFC: {afc_original:.3f}, AUC: {auc_original:.3f}, d': {dprime_original:.3f}")
    
    return {
        'maple': {'afc': afc_maple, 'auc': auc_maple, 'dprime': dprime_maple},
        'original': {'afc': afc_original, 'auc': auc_original, 'dprime': dprime_original}
    }


def run_metrics_section(data_id=3):
    Z_umap, Z_umap_original, raw_X, y = load_results_replot(data_id)

    import torch
    
    if not torch.is_tensor(Z_umap):
        Z_umap = torch.from_numpy(Z_umap)
    if not torch.is_tensor(Z_umap_original):
        Z_umap_original = torch.from_numpy(Z_umap_original)
    if not torch.is_tensor(y):
        y = torch.from_numpy(y)

    dataset_name = experiments_list[data_id]
    
    results = []
    
    # MAPLE results
    maple_results = {
        'dataset': dataset_name,
        'method': 'MAPLE',
        'neighborhood_purity': compute_neighborhood_purity(Z_umap, y, n_neighbors=15),
        'distance_consistency': compute_distance_consistency(Z_umap, y),
        'local_label_trustworthiness': local_label_trustworthiness(Z_umap, y, k=20, return_local=True),
        'calinski_harabasz': validation_measure(Z_umap, y, measure="calinski_harabasz"),
        'davies_bouldin': validation_measure(Z_umap, y, measure="davies_bouldin"),
        'hausdorff_distance': hausdorff_metric(Z_umap, y)
    }
    
    # Original UMAP results
    original_results = {
        'dataset': dataset_name,
        'method': 'Original',
        'neighborhood_purity': compute_neighborhood_purity(Z_umap_original, y, n_neighbors=15),
        'distance_consistency': compute_distance_consistency(Z_umap_original, y),
        # 'silhouette': validation_measure(Z_umap_original, y, measure="silhouette"),
        'local_label_trustworthiness': local_label_trustworthiness(Z_umap_original, y, k=20, return_local=True),
        'calinski_harabasz': validation_measure(Z_umap_original, y, measure="calinski_harabasz"),
        'davies_bouldin': validation_measure(Z_umap_original, y, measure="davies_bouldin"),
        'hausdorff_distance': hausdorff_metric(Z_umap_original, y)
    }
    
    # Add clustering metrics
    clustering_methods = ["kmeans", "dbscan"]
    clustering_metrics = ["arand", "ami", "nmi", "vmeasure"]
    
    for clustering_method in clustering_methods:
        for clustering_metric in clustering_metrics:
            if clustering_method == "kmeans":
                clustering_args = {"n_clusters": 10}
            elif clustering_method == "dbscan":
                clustering_args = {"eps": 0.5, "min_samples": 5}
            
            maple_results[f'{clustering_method}_{clustering_metric}'] = clustering_based_metrics(
                Z_umap, y, measure=clustering_metric, clustering=clustering_method, clustering_args=clustering_args)
            original_results[f'{clustering_method}_{clustering_metric}'] = clustering_based_metrics(
                Z_umap_original, y, measure=clustering_metric, clustering=clustering_method, clustering_args=clustering_args)
    

    maple_results['afc_accuracy'] = afc_accuracy(Z_umap.cpu().numpy(), y.cpu().numpy())
    original_results['afc_accuracy'] = afc_accuracy(Z_umap_original.cpu().numpy(), y.cpu().numpy())
    
    auc_maple, dprime_maple = pairwise_auc_and_dprime(Z_umap.cpu().numpy(), y.cpu().numpy())
    auc_original, dprime_original = pairwise_auc_and_dprime(Z_umap_original.cpu().numpy(), y.cpu().numpy())
    
    maple_results['pairwise_auc'] = auc_maple
    maple_results['pairwise_dprime'] = dprime_maple
    original_results['pairwise_auc'] = auc_original
    original_results['pairwise_dprime'] = dprime_original
    
    results.extend([maple_results, original_results])
    
    print(f"\n=== {dataset_name} ===")
    print(f"MAPLE - 2AFC: {maple_results['afc_accuracy']:.3f}, AUC: {auc_maple:.3f}, d': {dprime_maple:.3f}")
    print(f"Original - 2AFC: {original_results['afc_accuracy']:.3f}, AUC: {auc_original:.3f}, d': {dprime_original:.3f}")

    return results


def save_results_to_csv():
    import pandas as pd
    import os
    
    all_results = []
    
    for data_id in range(len(experiments_list)):
        print(f"Running metrics for {experiments_list[data_id]}")
        results = run_metrics_section(data_id)
        all_results.extend(results)
        print("\n--------------------------------")
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/metrics_results_csv_{len(experiments_list)}.csv", index=False)
    print(f"\nResults saved to: results/metrics_results_{len(experiments_list)}.csv")



if __name__ == "__main__":
    save_results_to_csv()

    # for data_id in range(len(experiments_list)):
    #     results = run_metrics_section(data_id)
    #     print("\n--------------------------------")


    # Z_umap, Z_umap_original, raw_X, y = load_results_replot(0)
    # perceptual_similarity_metrics(Z_umap, Z_umap_original, y)
