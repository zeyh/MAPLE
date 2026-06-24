import torch


def save_embedding_comparison_results( Z_umap_cosine, Z_tsne, raw_X, y, 
                                     dataset_name="unknown_dataset", dataset_size="unknown_size",
                                     n_neighbors=15, base_save_dir="results", compare_name = "embedding_comparison"):
    import os
    import json
    import numpy as np
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{dataset_name}_{dataset_size}_{compare_name}_{timestamp}"
    results_dir = os.path.join(base_save_dir, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, "Z_umap_cosine.npy"), Z_umap_cosine)
    np.save(os.path.join(results_dir, "Z_tsne.npy"), Z_tsne)
    np.save(os.path.join(results_dir, "raw_X.npy"), raw_X)
    np.save(os.path.join(results_dir, "y.npy"), y)
    
    config_summary = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "n_neighbors": n_neighbors,
        "embedding_methods": ["umap_proposed", "umap_cosine", "tsne"]
    }
    
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    try:
        from visualization.embedding_plot import vis_2_embedings
        
        cosine_tsne_path = os.path.join(results_dir, "umap_cosine_vs_tsne.png")
        vis_2_embedings(
            embeddings1=Z_umap_cosine,
            embeddings2=Z_tsne,
            labels=y,
            title1=f"UMAP Cosine ({dataset_name}, k={n_neighbors})",
            title2=f"t-SNE ({dataset_name})",
            filename=cosine_tsne_path,
            need_dr=False,
            is_saving=True,
            n_neighbors=n_neighbors
        )
    except ImportError:
        pass
    
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"SUMMARY\n")
        f.write(f"="*50 + "\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset_name} (size: {dataset_size})\n")
        f.write(f"n_neighbors: {n_neighbors}\n")
      
    
    mapping_info = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "size": dataset_size,
        "n_neighbors": n_neighbors,
        "timestamp": timestamp,
        "results_dir": results_dir,
        "experiment_type": "embedding_comparison"
    }
    
    registry_path = os.path.join(base_save_dir, "experiment_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    
    registry[experiment_id] = mapping_info
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return experiment_id, results_dir

def save_experiment_results(Z_umap, Z_umap_original, raw_X, y, results=None, 
                             DATASET_CONFIG=None, CONFIG_SET0=None, CONFIG_SET1=None, CONFIG_SET2=None, CONFIG_BT=None, CONFIG_MMCR=None,
                             latest_embedding_neighbors=None, fuzzy_set=None, pr_array=None, r_array=None,
                             geodesic_correlation=None, final_knn_indices=None, base_save_dir="results"):
    import os
    import json
    import numpy as np
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if DATASET_CONFIG is None:
        dataset_name = "unknown_dataset"
        dataset_size = "unknown_size"
        n_neighbors = "unknown_k"
    else:
        dataset_name = DATASET_CONFIG["name"]
        dataset_size = DATASET_CONFIG["size"]
        n_neighbors = CONFIG_SET1["n_neighbors"] if CONFIG_SET1 else "unknown_k"
    
    experiment_id = f"{dataset_name}_{dataset_size}_{n_neighbors}_{timestamp}"
    results_dir = os.path.join(base_save_dir, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, "Z_umap_proposed.npy"), Z_umap)
    np.save(os.path.join(results_dir, "Z_umap_original.npy"), Z_umap_original)
    np.save(os.path.join(results_dir, "raw_X.npy"), raw_X)
    np.save(os.path.join(results_dir, "y.npy"), y)
    if latest_embedding_neighbors is not None:
        if torch.is_tensor(latest_embedding_neighbors):
            latest_embedding_neighbors = latest_embedding_neighbors.cpu().numpy()
        np.save(os.path.join(results_dir, "latest_embedding_neighbors.npy"), latest_embedding_neighbors)
    
    if fuzzy_set is not None:
        from scipy.sparse import save_npz
        save_npz(os.path.join(results_dir, "fuzzy_set.npz"), fuzzy_set)
    
    if pr_array is not None:
        if torch.is_tensor(pr_array):
            pr_array = pr_array.cpu().numpy()
        np.save(os.path.join(results_dir, "pr_array.npy"), pr_array)
    
    if r_array is not None:
        if torch.is_tensor(r_array):
            r_array = r_array.cpu().numpy()
        np.save(os.path.join(results_dir, "r_array.npy"), r_array)
    
    if geodesic_correlation is not None:
        geodesic_path = os.path.join(results_dir, "geodesic_correlation.json")
        with open(geodesic_path, 'w') as f:
            json.dump(geodesic_correlation, f, indent=2)
    
    if final_knn_indices is not None:
        if torch.is_tensor(final_knn_indices):
            final_knn_indices = final_knn_indices.cpu().numpy()
        np.save(os.path.join(results_dir, "final_knn_indices.npy"), final_knn_indices)
    
    if all([DATASET_CONFIG, CONFIG_SET0, CONFIG_SET1, CONFIG_SET2, CONFIG_BT, CONFIG_MMCR]):
        config_summary = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "DATASET_CONFIG": DATASET_CONFIG,
            "CONFIG_SET0": CONFIG_SET0,
            "CONFIG_SET1": CONFIG_SET1, 
            "CONFIG_SET2": CONFIG_SET2,
            "CONFIG_BT": CONFIG_BT,
            "CONFIG_MMCR": CONFIG_MMCR
        }
        if geodesic_correlation is not None:
            config_summary["geodesic_correlation"] = geodesic_correlation
        
        config_path = os.path.join(results_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_summary, f, indent=2)

    try:
        from visualization.embedding_plot import vis_2_embedings
        proposed_title = f"Proposed ({dataset_name}, k={n_neighbors})"
        original_title = f"Original ({dataset_name}, k={n_neighbors})"
        fig_path = os.path.join(results_dir, "embedding_comparison.png")
        
        vis_2_embedings(
            embeddings1=Z_umap_original,
            embeddings2=Z_umap,
            labels=y,
            title1=original_title,
            title2=proposed_title,
            filename=fig_path,
            need_dr=False,
            is_saving=True,
            n_neighbors=n_neighbors
        )
        
    except ImportError:
        pass

    mapping_info = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "size": dataset_size,
        "n_neighbors": n_neighbors,
        "timestamp": timestamp,
        "results_dir": results_dir,
        "has_evaluation": results is not None
    }
    
    registry_path = os.path.join(base_save_dir, "experiment_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    
    registry[experiment_id] = mapping_info
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return experiment_id, results_dir


def save_experiment_results_minimal(Z_umap, Z_umap_original, y, 
                                     dataset_name="unknown", dataset_size="unknown", n_neighbors="unknown",
                                     base_save_dir="results"):
    import os
    import numpy as np
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{dataset_name}_{dataset_size}_{n_neighbors}_{timestamp}"
    results_dir = os.path.join(base_save_dir, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, "Z_umap_proposed.npy"), Z_umap)
    np.save(os.path.join(results_dir, "Z_umap_original.npy"), Z_umap_original)
    
    try:
        from visualization.embedding_plot import vis_2_embedings
        proposed_title = f"Proposed ({dataset_name}, k={n_neighbors})"
        original_title = f"Original ({dataset_name}, k={n_neighbors})"
        fig_path = os.path.join(results_dir, "embedding_comparison.png")
        
        vis_2_embedings(
            embeddings1=Z_umap_original,
            embeddings2=Z_umap,
            labels=y,
            title1=original_title,
            title2=proposed_title,
            filename=fig_path,
            need_dr=False,
            is_saving=True,
            n_neighbors=n_neighbors
        )
    except ImportError:
        pass
    
    return experiment_id, results_dir


def load_experiment_for_evaluation(experiment_id, base_save_dir="results"):
    import os
    import json
    import numpy as np
    
    experiment_dir = os.path.join(base_save_dir, experiment_id)
    
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    Z_umap = np.load(os.path.join(experiment_dir, "Z_umap_proposed.npy"))
    Z_umap_original = np.load(os.path.join(experiment_dir, "Z_umap_original.npy"))
    raw_X = np.load(os.path.join(experiment_dir, "raw_X.npy"))
    y = np.load(os.path.join(experiment_dir, "y.npy"))
    
    config_path = os.path.join(experiment_dir, "config.json")
    configs = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            configs = json.load(f)
    
    results_path = os.path.join(experiment_dir, "zadu_results.json")
    existing_results = None
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
    
    return {
        "Z_umap": Z_umap,
        "Z_umap_original": Z_umap_original,
        "raw_X": raw_X,
        "y": y,
        "configs": configs,
        "existing_results": existing_results,
        "experiment_dir": experiment_dir
    }


def get_trained_embeddings(model, X: torch.Tensor, ):
    from config.default_config import DEVICE

    model.eval()
    with torch.no_grad():
        X = X.to(DEVICE) 
        embeddings = model(X)
        embeddings = embeddings.cpu()
    
    return embeddings


def save_dr_results(Z_umap, Z_umap_original, y, save_dir, dataset_name):
    import os
    import numpy as np
    from datetime import datetime
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    np.save(os.path.join(save_dir, f"Z_umap_{timestamp}_{dataset_name}.npy"), Z_umap.cpu().numpy() if torch.is_tensor(Z_umap) else Z_umap)
    np.save(os.path.join(save_dir, f"Z_umap_original_{timestamp}_{dataset_name}.npy"), Z_umap_original.cpu().numpy() if torch.is_tensor(Z_umap_original) else Z_umap_original)
    np.save(os.path.join(save_dir, f"labels_{timestamp}_{dataset_name}.npy"), y.cpu().numpy() if torch.is_tensor(y) else y)
    
    return timestamp

def load_embeddings(timestamp, save_dir="saved_embeddings"):
    import os
    import numpy as np
    
    Z_umap = np.load(os.path.join(save_dir, f"Z_umap_{timestamp}.npy"))
    Z_umap_original = np.load(os.path.join(save_dir, f"Z_umap_original_{timestamp}.npy"))
    y = np.load(os.path.join(save_dir, f"labels_{timestamp}.npy"))
    
    return Z_umap, Z_umap_original, y



if __name__ == "__main__":
    import numpy as np
    import os
    dir = "results/fashion_mnist_full_40000_15_20251224_130944"
    r_name = "r_array.npy"
    pr_name = "pr_array.npy"
    r = np.load(os.path.join(dir, r_name))
    pr = np.load(os.path.join(dir, pr_name))
    print(f"r shape: {r.shape}")
    print(f"pr shape: {pr.shape}")
    print(f"r: {r}")
    print(f"pr: {pr}")