import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import json
from datetime import datetime
from src_v2.data.toy_data import loading_toy_data
from src_v2.evaluation.running_time import maple_with_time, print_timing_results
from src_v2.training.training_new import compute_knn_similarity
from src_v2.training import training_with_bt_dynamic_knn
from src_v2.utils.umap_utils import build_knn_graph
from src_v2.config import set_test_config

# experiments_list = [
#     "STL_10_13000_15_20250827_153823", 
#     "mnist_test_10000_15_20250826_152503", 
#     "fashion_mnist_full_40000_15_20250828_010753",
#     "celegans_l2_15000_30_20250829_002309", 
#     "celegans_l2_neuron_12029_20_20250901_143716",
#     "celegans_l2_neuron_12029_20_20250831_013202",
#     "celegans_l2_glia_3259_20_20250831_010032",
#     "celegans_processed_6188_20_20250902_134946",
#     "scp2745_4000_20_20250831_172917",
# ]

datasets_list = [
    # "mnist_test",
    # "fashion_mnist_full",
    # "STL_10",
    # "pbmc3k",
    # "celegans_l2_neuron",
    # "celegans_l2_glia",
    # "celegans_processed",
    "celegans_l2",
]

DEFAULT_K = 10
DEFAULT_LAMBDA = 0.1
DEFAULT_UPDATE = 9999
DEFAULT_EPOCHS = 20
DEFAULT_BT = "False"


def parse_experiment_id(exp_id):
    parts = exp_id.split('_')
    
    dataset_name_parts = []
    size = None
    k = None
    
    for i, part in enumerate(parts):
        if part.isdigit():
            if size is None:
                size = int(part)
            elif k is None:
                k = int(part)
                dataset_name = '_'.join(parts[:i])
                break
    
    if size is None or k is None:
        raise ValueError(f"Cannot parse experiment ID: {exp_id}")
    
    return {
        "dataset_name": dataset_name,
        "size": size,
        "k": k
    }


def load_dataset(dataset_name, dataset_size):
    X, y = loading_toy_data(dataset_name, number=dataset_size)
    
    if isinstance(X, torch.Tensor):
        X = X.float()
    else:
        X = torch.tensor(X, dtype=torch.float32)
    
    if isinstance(y, torch.Tensor):
        y = y.long()
    else:
        y = torch.tensor(y, dtype=torch.long)
    
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    
    return X, y


def run_jaccard_experiment(exp_id):
    parsed = parse_experiment_id(exp_id)
    dataset_name = parsed["dataset_name"]
    dataset_size = parsed["size"]
    n_neighbors = parsed["k"]
    
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    
    try:
        X, y = load_dataset(dataset_name, dataset_size)
        
        feature_dim = X.shape[1]
        n_samples = X.shape[0]
        
        batch_size = 4096
        if feature_dim > 10000:
            manifold_size_gb = (n_samples * (n_neighbors + 1) * feature_dim * 4) / (1024**3)
            if manifold_size_gb > 8.0:
                batch_size = max(512, int(4096 * (8.0 / manifold_size_gb)))
                print(f"High-dimensional data detected (D={feature_dim}). Reducing batch_size to {batch_size}")
        
        test_params = {
            "batch_size": batch_size,
            "Projection_dim": 128,
            "n_neighbors": n_neighbors,
            "lmbda": DEFAULT_LAMBDA,
            "epochs": DEFAULT_EPOCHS,
            "update_knn_every": DEFAULT_UPDATE,
            "use_bt_loss": DEFAULT_BT,
            "use_amp": True,
            "use_dimension_shift": False,
            "shift_warmup_epochs": 5,
        }
        set_test_config(test_params)
        
        print(f"\n{'='*80}")
        print(f"Step 1: Building initial KNN graph...")
        print(f"{'='*80}")
        raw_knn_indices, _, _ = build_knn_graph(X, n_neighbors)
        
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        timing_dict, (Z_umap, Z_umap_original, y_returned) = maple_with_time(
            X, y,
            save_results=True,
            skip_eval=False,
            n_neighbors=n_neighbors
        )
        
        print(f"\n{'='*80}")
        print(f"Step 3: Getting final KNN indices for Jaccard computation...")
        print(f"{'='*80}")
        _, _, _, _, _, final_knn_indices = training_with_bt_dynamic_knn(X, raw_knn_indices)
        
        print(f"\n{'='*80}")
        print(f"Step 4: Computing Jaccard similarity...")
        print(f"{'='*80}")
        jaccard_sim, overlap_ratio = compute_knn_similarity(raw_knn_indices, final_knn_indices)
        
        print(f"Jaccard Similarity: {jaccard_sim:.4f} ({jaccard_sim:.2%})")
        print(f"Overlap Ratio: {overlap_ratio:.4f} ({overlap_ratio:.2%})")
        
        result = {
            "experiment_id": exp_id,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "n_neighbors": n_neighbors,
            "jaccard_similarity": float(jaccard_sim),
            "overlap_ratio": float(overlap_ratio),
            "timing": timing_dict,
            "metrics": {
                "purity_original": timing_dict.get("purity_original"),
                "purity_inferred": timing_dict.get("purity_inferred"),
            },
            "error": None
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return {
            "experiment_id": exp_id,
            "dataset_name": parsed.get("dataset_name", "unknown"),
            "dataset_size": parsed.get("size", "unknown"),
            "n_neighbors": parsed.get("k", "unknown"),
            "jaccard_similarity": None,
            "overlap_ratio": None,
            "timing": None,
            "metrics": None,
            "error": str(e)
        }


def save_results(results, timestamp):
    filename = f"jaccard_results_{timestamp}.json"
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiments": results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    return filename


def test_data_loading(datasets=None, test_sizes=None):
    if datasets is None:
        datasets = datasets_list
    
    if test_sizes is None:
        test_sizes = [1000]
    
    print(f"\n{'#'*80}")
    print(f"# Test sizes: {test_sizes}")
    print(f"{'#'*80}")
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        
        dataset_results = []
        
        for test_size in test_sizes:
            print(f"\n  Testing with size={test_size}...")
            try:
                X, y = load_dataset(dataset_name, test_size)
                
                dataset_results.append({
                    "size": test_size,
                    "success": True,
                    "X_shape": list(X.shape),
                    "y_shape": list(y.shape),
                    "n_classes": len(torch.unique(y)),
                    "feature_dim": X.shape[1],
                    "error": None
                })
                
                print(f"  Success: X{X.shape}, y{y.shape}, {len(torch.unique(y))} classes")
                
            except Exception as e:
                print(f"  Failed: {e}")
                import traceback
                traceback.print_exc()
                
                dataset_results.append({
                    "size": test_size,
                    "success": False,
                    "X_shape": None,
                    "y_shape": None,
                    "n_classes": None,
                    "feature_dim": None,
                    "error": str(e)
                })
        
        results[dataset_name] = dataset_results
        
        successful_tests = sum(1 for r in dataset_results if r["success"])
    
    print(f"\n{'#'*80}")
    print(f"# DATA LOADING TEST COMPLETE")
    print(f"{'#'*80}")
    
    total_datasets = len(datasets)
    successful_datasets = sum(1 for r in results.values() if any(test["success"] for test in r))
    
    print(f"\nSummary:")
    print(f"  Total datasets tested: {total_datasets}")
    print(f"  Datasets with at least one successful load: {successful_datasets}")
    
    return results


def run_jaccard_experiment_for_dataset(dataset_name, dataset_size, n_neighbors):
    print(f"{'='*80}")
    
    try:
        X, y = load_dataset(dataset_name, dataset_size)
        
        feature_dim = X.shape[1]
        n_samples = X.shape[0]
        
        batch_size = 4096
        if feature_dim > 10000:
            manifold_size_gb = (n_samples * (n_neighbors + 1) * feature_dim * 4) / (1024**3)
            if manifold_size_gb > 8.0:
                batch_size = max(512, int(4096 * (8.0 / manifold_size_gb)))
                print(f"High-dimensional data detected (D={feature_dim}). Reducing batch_size to {batch_size}")
        
        test_params = {
            "batch_size": batch_size,
            "Projection_dim": 128,
            "n_neighbors": n_neighbors,
            "lmbda": DEFAULT_LAMBDA,
            "epochs": DEFAULT_EPOCHS,
            "update_knn_every": DEFAULT_UPDATE,
            "use_bt_loss": DEFAULT_BT,
            "use_amp": True,
            "use_dimension_shift": False,
            "shift_warmup_epochs": 5,
        }
        set_test_config(test_params)
        
        print(f"\n{'='*80}")
        print(f"Step 1: Building initial KNN graph...")
        print(f"{'='*80}")
        raw_knn_indices, _, _ = build_knn_graph(X, n_neighbors)
        
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        timing_dict, (Z_umap, Z_umap_original, y_returned) = maple_with_time(
            X, y,
            save_results=True,
            skip_eval=False,
            n_neighbors=n_neighbors
        )
        
        print(f"\n{'='*80}")
        print(f"Step 3: Getting final KNN indices for Jaccard computation...")
        print(f"{'='*80}")
        _, _, _, _, _, final_knn_indices = training_with_bt_dynamic_knn(X, raw_knn_indices)
        
        print(f"\n{'='*80}")
        print(f"Step 4: Computing Jaccard similarity...")
        print(f"{'='*80}")
        jaccard_sim, overlap_ratio = compute_knn_similarity(raw_knn_indices, final_knn_indices)
        
        print(f"Jaccard Similarity: {jaccard_sim:.4f} ({jaccard_sim:.2%})")
        print(f"Overlap Ratio: {overlap_ratio:.4f} ({overlap_ratio:.2%})")
        
        result = {
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "n_neighbors": n_neighbors,
            "jaccard_similarity": float(jaccard_sim),
            "overlap_ratio": float(overlap_ratio),
            "timing": timing_dict,
            "metrics": {
                "purity_original": timing_dict.get("purity_original"),
                "purity_inferred": timing_dict.get("purity_inferred"),
            },
            "error": None
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return {
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "n_neighbors": n_neighbors,
            "jaccard_similarity": None,
            "overlap_ratio": None,
            "timing": None,
            "metrics": None,
            "error": str(e)
        }


def run_all_jaccard_experiments(datasets=None, default_size=5000, default_k=None):
    if datasets is None:
        datasets = datasets_list
    
    if default_k is None:
        default_k = DEFAULT_K
    
    print(f"\n{'#'*80}")
    print(f"# Default size: {default_size}, Default k: {default_k}")
    print(f"{'#'*80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")
        
        result = run_jaccard_experiment_for_dataset(dataset_name, default_size, default_k)
        all_results.append(result)
        
        save_results(all_results, timestamp)
        
        if result.get("error") is None:
    
    successful = sum(1 for r in all_results if r.get("error") is None)
    failed = len(all_results) - successful
    print(f"\nSummary: {successful} successful, {failed} failed")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if True: #len(sys.argv) > 1 and sys.argv[1] == "test":
        test_data_loading()
    else:   
        run_all_jaccard_experiments()   
