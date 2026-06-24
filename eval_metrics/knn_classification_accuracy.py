import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# global_dir =  "results/fashion_mnist_full_40000_30_20260111_141610" # fashion_mnist_full_40000_15_20260111_122903 "results/fashion_mnist_full_40000_15_20251224_130944"
# global_dir = "results/mnist_test_10000_15_20260111_110920"
global_dir = "results/fashion_mnist_full_40000_15_20250828_010753"


def load_knn_indices(directory = None):
    if directory is None:
        directory = global_dir
    knn_indices = np.load(os.path.join(directory, "final_knn_indices.npy"))
    return knn_indices

def load_label_y(directory = None):
    if directory is None:
        directory = global_dir
    label_y = np.load(os.path.join(directory, "y.npy"))
    return label_y

def perform_knn_classification_accuracy(directory=None, method='loo', k_values=[14], 
                                        use_weighted_vote=False, test_size=0.2, n_folds=5, 
                                        random_state=42, report_classwise=True):
    if directory is None:
        directory = global_dir
    
    if method == 'train_test':
        if use_weighted_vote:
            use_weighted_vote = False
    
    raw_X = load_raw_data(directory)
    label_y = load_label_y(directory)
    
    # Load learned KNN indices
    try:
        learned_knn_indices = load_knn_indices(directory)
        if learned_knn_indices.shape[0] != len(label_y):
            raise ValueError(f"KNN indices shape mismatch: {learned_knn_indices.shape[0]} != {len(label_y)}")
        if learned_knn_indices.shape[1] < 2:
            raise ValueError(f"KNN indices must have at least 2 columns (self + 1 neighbor), got {learned_knn_indices.shape[1]}")
    except FileNotFoundError:
        learned_knn_indices = None
    except Exception as e:
        learned_knn_indices = None
    
    # Compute original KNN indices
    # Use max k to ensure we have enough neighbors for all k values being tested
    n_neighbors = max(k_values) if k_values else 14
    print(f"Computing original KNN graph (k={n_neighbors}, metric='cosine')...")
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine", n_jobs=-1)
    knn.fit(raw_X)
    original_knn_indices = knn.kneighbors(raw_X, return_distance=False)
    # Ensure format: add self as first column if not present
    if original_knn_indices[0, 0] != 0:
        original_knn_indices = np.column_stack([np.arange(len(raw_X)), original_knn_indices])
    
    # Load fuzzy_set for weighted voting if available (only safe in LOO mode)
    edge_weights = None
    if use_weighted_vote:
        try:
            from scipy.sparse import load_npz
            fuzzy_set_path = os.path.join(directory, "fuzzy_set.npz")
            if os.path.exists(fuzzy_set_path):
                fuzzy_set = load_npz(fuzzy_set_path)
                edge_weights = fuzzy_set.tocsr()
                print(" Loaded fuzzy_set for weighted voting")
            else:
                print("  fuzzy_set.npz not found, using uniform voting")
                use_weighted_vote = False
        except Exception as e:
            print(f"  Could not load fuzzy_set: {e}, using uniform voting")
            use_weighted_vote = False
    
    results = {}
    
    # Evaluate original graph
    print(f"\n{'='*60}")
    print(f"Evaluating Original Graph (method={method})")
    print(f"{'='*60}")
    original_results = evaluate_knn_classification(
        knn_indices=original_knn_indices,
        labels=label_y,
        method=method,
        k_values=k_values,
        use_weighted_vote=False,  
        edge_weights=None,
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state
    )
    results['original'] = original_results
    
    # Evaluate learned graph
    if learned_knn_indices is not None:
        print(f"\n{'='*60}")
        print(f"Evaluating Learned Graph (method={method})")
        print(f"{'='*60}")
        learned_results = evaluate_knn_classification(
            knn_indices=learned_knn_indices,
            labels=label_y,
            method=method,
            k_values=k_values,
            use_weighted_vote=use_weighted_vote,
            edge_weights=edge_weights,
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state
        )
        results['learned'] = learned_results
        
        for k in k_values:
            orig_acc = original_results.get(f'accuracy_k{k}', 0.0)
            learned_acc = learned_results.get(f'accuracy_k{k}', 0.0)
            improvement = learned_acc - orig_acc
            improvement_pct = (improvement / orig_acc * 100) if orig_acc > 0 else 0.0
            
            orig_skipped = original_results.get(f'skipped_points_k{k}', 0)
            learned_skipped = learned_results.get(f'skipped_points_k{k}', 0)
            orig_skipped_pct = original_results.get(f'skipped_pct_k{k}', 0.0)
            learned_skipped_pct = learned_results.get(f'skipped_pct_k{k}', 0.0)
            
            print(f"k={k:2d}: Original={orig_acc:.4f}, Learned={learned_acc:.4f}, "
                  f"Improvement={improvement:+.4f} ({improvement_pct:+.2f}%)")
            
            if orig_skipped > 0 or learned_skipped > 0:
                print(f"      Skipped: Original={orig_skipped} ({orig_skipped_pct:.2f}%), "
                      f"Learned={learned_skipped} ({learned_skipped_pct:.2f}%)")
        
        if method == 'loo' and report_classwise:
            class_names = None
            n_classes = len(np.unique(label_y))
            if n_classes == 10:
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            for k in k_values:
                if k <= original_knn_indices.shape[1] - 1 and k <= learned_knn_indices.shape[1] - 1:
                    classwise_results = report_classwise_comparison(
                        original_knn_indices=original_knn_indices,
                        learned_knn_indices=learned_knn_indices,
                        labels=label_y,
                        k=k,
                        original_edge_weights=None,
                        learned_edge_weights=edge_weights,  # Not used 
                        use_weighted_vote=use_weighted_vote,  # Not used 
                        class_names=class_names
                    )
                    if 'classwise' not in results:
                        results['classwise'] = {}
                    results['classwise'][f'k{k}'] = classwise_results
        
        # Analyze skipped points 
        if method == 'loo' and learned_knn_indices is not None:
            z_2d = None
            try:
                z_2d_path = os.path.join(directory, "Z_umap_proposed.npy")
                if os.path.exists(z_2d_path):
                    z_2d = np.load(z_2d_path)
                    if z_2d.shape[1] > 2:
                        z_2d = z_2d[:, :2]
            except Exception as e:
                print(f"  Could not load 2D embedding: {e}, skipping spatial analysis")
            
            # Get class names   
            class_names = None
            n_classes = len(np.unique(label_y))
            if n_classes == 10:
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
            for k in k_values:
                if k <= original_knn_indices.shape[1] - 1 and k <= learned_knn_indices.shape[1] - 1:
                    original_skipped_analysis = analyze_skipped_points(
                        knn_indices=original_knn_indices,
                        labels=label_y,
                        k=k,
                        use_weighted_vote=False,
                        edge_weights=None,
                        z_2d=z_2d,
                        class_names=class_names
                    )
                    
                    learned_skipped_analysis = analyze_skipped_points(
                        knn_indices=learned_knn_indices,
                        labels=label_y,
                        k=k,
                        use_weighted_vote=use_weighted_vote,
                        edge_weights=edge_weights,
                        z_2d=z_2d,
                        class_names=class_names
                    )
                    
                    report_skipped_points_analysis(
                        original_analysis=original_skipped_analysis,
                        learned_analysis=learned_skipped_analysis,
                        labels=label_y,
                        class_names=class_names
                    )
                    
                    if 'skipped_analysis' not in results:
                        results['skipped_analysis'] = {}
                    results['skipped_analysis'][f'k{k}'] = {
                        'original': original_skipped_analysis,
                        'learned': learned_skipped_analysis
                    }
    
    return results


def evaluate_knn_classification(knn_indices, labels, method='loo', k_values=[14],
                                use_weighted_vote=False, edge_weights=None,
                                test_size=0.2, n_folds=5, random_state=42):
    N = len(labels)
    
    if knn_indices.shape[0] != N:
        raise ValueError(f"KNN indices shape mismatch: {knn_indices.shape[0]} != {N}")
    if knn_indices.shape[1] < 2:
        raise ValueError(f"KNN indices must have at least 2 columns (self + 1 neighbor)")
    
    sample_size = min(100, N)
    for i in range(sample_size):
        if knn_indices[i, 0] != i:
            raise ValueError(f"KNN indices format error: row {i} has {knn_indices[i, 0]} in first column (expected {i})")
    
    results = {}
    
    if method == 'loo':
        for k in k_values:
            max_available_k = knn_indices.shape[1] - 1
            if k > max_available_k:
                continue
            
            correct_predictions = 0
            total_predictions = 0
            skipped_points = 0
            skipped_indices_list = []
            
            for i in tqdm(range(N), desc=f"k={k}"):
                neighbor_indices = knn_indices[i, 1:k+1]
                valid_mask = (neighbor_indices >= 0) & (neighbor_indices < N)
                neighbor_indices = neighbor_indices[valid_mask]
                
                if len(neighbor_indices) == 0:
                    skipped_points += 1
                    skipped_indices_list.append(i)
                    continue
                
                neighbor_labels = labels[neighbor_indices]
                
                # Predict label
                is_skipped = False
                if use_weighted_vote and edge_weights is not None:
                    # Weighted voting using edge weights
                    weights = []
                    valid_neighbors = []
                    # Get row from sparse matrix
                    row = edge_weights.getrow(i)
                    row_dict = {col: val for col, val in zip(row.indices, row.data)}
                    for neighbor_idx in neighbor_indices:
                        weight_val = row_dict.get(neighbor_idx, 0.0)
                        if weight_val > 0:
                            weights.append(float(weight_val))
                            valid_neighbors.append(labels[neighbor_idx])
                    
                    if len(valid_neighbors) == 0:
                        skipped_points += 1
                        skipped_indices_list.append(i)
                        is_skipped = True
                    
                    if not is_skipped:
                        weights = np.array(weights)
                        valid_neighbors = np.array(valid_neighbors)
                        
                        # Weighted majority vote
                        unique_labels = np.unique(valid_neighbors)
                        weighted_votes = {}
                        for label in unique_labels:
                            mask = valid_neighbors == label
                            weighted_votes[label] = weights[mask].sum()
                        
                        predicted_label = max(weighted_votes, key=weighted_votes.get)
                else:
                    # Uniform majority vote
                    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                    predicted_label = unique_labels[np.argmax(counts)]
                
                if not is_skipped:
                    if predicted_label == labels[i]:
                        correct_predictions += 1
                    total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            results[f'accuracy_k{k}'] = accuracy
            results[f'skipped_points_k{k}'] = skipped_points
            skipped_pct = (skipped_points / N * 100) if N > 0 else 0.0
            results[f'skipped_indices_k{k}'] = np.array(skipped_indices_list)
            
            print(f"  Accuracy@k={k}: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
            if skipped_points > 0:
                print(f"  Skipped points: {skipped_points} ({skipped_pct:.2f}% of dataset)")
                print(f"        Accuracy reported only on classifiable points ({total_predictions}/{N})")
            results[f'skipped_pct_k{k}'] = skipped_pct
    
    elif method == 'train_test':
        from sklearn.model_selection import train_test_split
        
        for k in k_values:
            max_available_k = knn_indices.shape[1] - 1
            if k > max_available_k:
                print(f"  k={k} exceeds available neighbors (max={max_available_k}), skipping")
                continue
            
            accuracies = []
            all_skipped_counts = []
            
            print(f"Computing train/test accuracy for k={k} (n_folds={n_folds})...")
            
            for fold in range(n_folds):
                train_indices, test_indices = train_test_split(
                    np.arange(N), test_size=test_size, random_state=random_state + fold, 
                    stratify=labels
                )
                
                train_labels = labels[train_indices]
                train_indices_set = set(train_indices)  
                
                correct_predictions = 0
                total_predictions = 0
                skipped_points = 0
                
                for test_idx in tqdm(test_indices, desc=f"Fold {fold+1}/{n_folds}", leave=False):
                    neighbor_indices_full = knn_indices[test_idx, 1:k+1]
                    neighbor_indices_train = [idx for idx in neighbor_indices_full if idx in train_indices_set]
                    neighbor_indices_train = [idx for idx in neighbor_indices_train if 0 <= idx < N]
                    
                    if len(neighbor_indices_train) == 0:
                        skipped_points += 1
                        continue
                    
                    neighbor_indices_train = neighbor_indices_train[:k]
                    neighbor_labels = labels[neighbor_indices_train]
                    
                    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                    predicted_label = unique_labels[np.argmax(counts)]
                    
                    if predicted_label == labels[test_idx]:
                        correct_predictions += 1
                    total_predictions += 1
                
                fold_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                accuracies.append(fold_accuracy)
                all_skipped_counts.append(skipped_points)
            
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            mean_skipped = np.mean(all_skipped_counts)
            mean_skipped_pct = (mean_skipped / (N * test_size) * 100) if N > 0 else 0.0
            
            results[f'accuracy_k{k}'] = mean_accuracy
            results[f'accuracy_k{k}_std'] = std_accuracy
            results[f'skipped_points_k{k}'] = int(mean_skipped)
            results[f'skipped_pct_k{k}'] = mean_skipped_pct
            
            print(f"  Accuracy@k={k}: {mean_accuracy:.4f} ± {std_accuracy:.4f} (mean ± std over {n_folds} folds)")
            if mean_skipped > 0:
                print(f"  Skipped test points (avg per fold): {mean_skipped:.1f} ({mean_skipped_pct:.2f}% of test set)")
    
    return results


def analyze_skipped_points(knn_indices, labels, k, use_weighted_vote=False, edge_weights=None,
                          z_2d=None, class_names=None):
    N = len(labels)
    skipped_indices = []
    
    for i in range(N):
        neighbor_indices = knn_indices[i, 1:k+1]
        valid_mask = (neighbor_indices >= 0) & (neighbor_indices < N)
        neighbor_indices = neighbor_indices[valid_mask]
        
        if len(neighbor_indices) == 0:
            skipped_indices.append(i)
            continue
        
        # For weighted voting, check if any neighbors have positive weights
        if use_weighted_vote and edge_weights is not None:
            row = edge_weights.getrow(i)
            row_dict = {col: val for col, val in zip(row.indices, row.data)}
            has_valid_weight = any(row_dict.get(nidx, 0.0) > 0 for nidx in neighbor_indices)
            if not has_valid_weight:
                skipped_indices.append(i)
    
    skipped_indices = np.array(skipped_indices)
    n_skipped = len(skipped_indices)
    
    if n_skipped == 0:
        return {
            'n_skipped': 0,
            'skipped_pct': 0.0,
            'skipped_indices': np.array([]),
            'label_distribution': {},
            'boundary_analysis': None
        }
    
    # Analyze label distribution of skipped points
    skipped_labels = labels[skipped_indices]
    unique_labels = np.unique(labels)
    label_counts = {}
    label_pcts = {}
    
    for label in unique_labels:
        count = np.sum(skipped_labels == label)
        total_count = np.sum(labels == label)
        label_counts[int(label)] = count
        label_pcts[int(label)] = (count / total_count * 100) if total_count > 0 else 0.0
    
    # Analyze spatial distribution in 2D
    boundary_analysis = None
    if z_2d is not None and z_2d.shape[0] == N and n_skipped > 0:
        # Compute distances to class boundaries
        # For each skipped point, find distance to nearest point of different class
        skipped_z = z_2d[skipped_indices]
        skipped_labels_array = labels[skipped_indices]
        
        # For each skipped point, compute min distance to different class
        min_distances_to_boundary = []
        for i, (z_point, label) in enumerate(zip(skipped_z, skipped_labels_array)):
            # Find all points with different labels
            different_class_mask = labels != label
            if np.any(different_class_mask):
                different_class_z = z_2d[different_class_mask]
                distances = cdist([z_point], different_class_z)[0]
                min_distances_to_boundary.append(np.min(distances))
            else:
                min_distances_to_boundary.append(np.inf)
        
        min_distances_to_boundary = np.array(min_distances_to_boundary)
        
        # Compare with average distance to boundary for all points
        sample_size = min(1000, N)
        np.random.seed(42) 
        sample_indices = np.random.choice(N, size=sample_size, replace=False)
        all_min_distances = []
        for i in sample_indices:
            z_point = z_2d[i]
            label = labels[i]
            different_class_mask = labels != label
            if np.any(different_class_mask):
                different_class_z = z_2d[different_class_mask]
                distances = cdist([z_point], different_class_z)[0]
                all_min_distances.append(np.min(distances))
        
        if len(all_min_distances) > 0 and len(min_distances_to_boundary[min_distances_to_boundary < np.inf]) > 0:
            avg_boundary_distance = np.mean(all_min_distances)
            skipped_avg_boundary_distance = np.mean(min_distances_to_boundary[min_distances_to_boundary < np.inf])
            
            boundary_analysis = {             # Points closer to boundary are more likely to be ambiguous
                'avg_distance_to_boundary_skipped': float(skipped_avg_boundary_distance),
                'avg_distance_to_boundary_all': float(avg_boundary_distance),
                'closer_to_boundary_ratio': float(skipped_avg_boundary_distance / avg_boundary_distance) if avg_boundary_distance > 0 else 1.0,
                'n_near_boundary': int(np.sum(min_distances_to_boundary < avg_boundary_distance * 0.5))
            }
    
    analysis = {
        'n_skipped': n_skipped,
        'skipped_pct': (n_skipped / N * 100) if N > 0 else 0.0,
        'skipped_indices': skipped_indices,
        'label_distribution': {
            'counts': label_counts,
            'percentages': label_pcts
        },
        'boundary_analysis': boundary_analysis
    }
    
    return analysis


def report_skipped_points_analysis(original_analysis, learned_analysis, labels, class_names=None):
    print(f"\n{'='*60}")
    print(f"Skipped Points Analysis")
    print(f"{'='*60}")
    print(f"\nOverall Statistics:")
    print(f"  Original graph: {original_analysis['n_skipped']} skipped ({original_analysis['skipped_pct']:.2f}%)")
    print(f"  Learned graph: {learned_analysis['n_skipped']} skipped ({learned_analysis['skipped_pct']:.2f}%)")
    
    # Label distribution analysis
    if learned_analysis['n_skipped'] > 0:
        print(f"\n{'='*60}")
        print(f"Label Distribution of Skipped Points (Learned Graph):")
        print(f"{'='*60}")
        print(f"{'Class':<20} {'Skipped':<12} {'% of Class':<15} {'% of All Skipped':<18}")
        print(f"{'-'*70}")
        
        learned_label_counts = learned_analysis['label_distribution']['counts']
        learned_label_pcts = learned_analysis['label_distribution']['percentages']
        total_skipped = learned_analysis['n_skipped']
        
        # Sort by percentage of class that was skipped
        sorted_by_pct = sorted(learned_label_pcts.items(), key=lambda x: x[1], reverse=True)
        
        for label, pct_of_class in sorted_by_pct:
            count = learned_label_counts[label]
            pct_of_skipped = (count / total_skipped * 100) if total_skipped > 0 else 0.0
            class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else f"Class {int(label)}"
            print(f"{class_name:<20} {count:<12} {pct_of_class:>14.2f}%  {pct_of_skipped:>17.2f}%")
        
        # Identify classes with highest skip rates
        top_skipped_classes = sorted_by_pct[:3]
        print(f"\nTop 3 Classes with Highest Skip Rates:")
        for label, pct in top_skipped_classes:
            class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else f"Class {int(label)}"
            print(f"  {class_name}: {pct:.2f}% of class points skipped")
    
    # Boundary analysis
    if learned_analysis['boundary_analysis'] is not None:
        print(f"\n{'='*60}")
        print(f"Spatial Analysis (2D Embedding):")
        print(f"{'='*60}")
        ba = learned_analysis['boundary_analysis']
        print(f"  Average distance to class boundary:")
        print(f"    All points: {ba['avg_distance_to_boundary_all']:.4f}")
        print(f"    Skipped points: {ba['avg_distance_to_boundary_skipped']:.4f}")
        print(f"    Ratio (skipped/all): {ba['closer_to_boundary_ratio']:.3f}")
        
        print(f"  Points near boundary (<50% of avg distance): {ba['n_near_boundary']}/{learned_analysis['n_skipped']}")


def compute_classwise_accuracy(knn_indices, labels, k, use_weighted_vote=False, edge_weights=None):
    N = len(labels)
    predictions = np.zeros(N, dtype=labels.dtype)
    
    for i in range(N):
        neighbor_indices = knn_indices[i, 1:k+1]
        valid_mask = (neighbor_indices >= 0) & (neighbor_indices < N)
        neighbor_indices = neighbor_indices[valid_mask]
        
        if len(neighbor_indices) == 0:
            predictions[i] = labels[np.argmax(np.bincount(labels))]
            continue
        
        neighbor_labels = labels[neighbor_indices]
        
        if use_weighted_vote and edge_weights is not None:
            weights = []
            valid_neighbors = []
            row = edge_weights.getrow(i)
            row_dict = {col: val for col, val in zip(row.indices, row.data)}
            for neighbor_idx in neighbor_indices:
                weight_val = row_dict.get(neighbor_idx, 0.0)
                if weight_val > 0:
                    weights.append(float(weight_val))
                    valid_neighbors.append(labels[neighbor_idx])
            
            if len(valid_neighbors) == 0:
                predictions[i] = labels[np.argmax(np.bincount(labels))]
                continue
            
            weights = np.array(weights)
            valid_neighbors = np.array(valid_neighbors)
            
            # Weighted majority vote
            unique_labels = np.unique(valid_neighbors)
            weighted_votes = {}
            for label in unique_labels:
                mask = valid_neighbors == label
                weighted_votes[label] = weights[mask].sum()
            
            predictions[i] = max(weighted_votes, key=weighted_votes.get)
        else:
            # Uniform majority vote
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predictions[i] = unique_labels[np.argmax(counts)]
    
    # confusion matrix
    unique_labels = np.unique(labels)
    confusion_mat = confusion_matrix(labels, predictions, labels=unique_labels)
    
    # per-class accuracy
    class_accuracies = {}
    for idx, label in enumerate(unique_labels):
        class_mask = labels == label
        if np.sum(class_mask) > 0:
            class_correct = np.sum((labels[class_mask] == predictions[class_mask]))
            class_total = np.sum(class_mask)
            class_accuracies[int(label)] = class_correct / class_total
    
    return predictions, class_accuracies, confusion_mat


def report_classwise_comparison(original_knn_indices, learned_knn_indices, labels, k,
                                original_edge_weights=None, learned_edge_weights=None,
                                use_weighted_vote=False, class_names=None):
    print(f"\n{'='*60}")
    print(f"Class-wise Accuracy Analysis (k={k})")
    print(f"{'='*60}")
    
    orig_predictions, orig_class_acc, orig_confusion = compute_classwise_accuracy(
        original_knn_indices, labels, k, use_weighted_vote=False, edge_weights=None
    )
    
    learned_predictions, learned_class_acc, learned_confusion = compute_classwise_accuracy(
        learned_knn_indices, labels, k, use_weighted_vote=False, edge_weights=None
    )
    
    unique_labels = np.unique(labels)
    unique_labels = sorted(unique_labels)
    
    print(f"\n{'Class':<15} {'Original':<12} {'Learned':<12} {'Improvement':<15} {'% Change':<12}")
    print(f"{'-'*70}")
    
    improvements = {}
    for label in unique_labels:
        orig_acc = orig_class_acc.get(int(label), 0.0)
        learned_acc = learned_class_acc.get(int(label), 0.0)
        improvement = learned_acc - orig_acc
        improvement_pct = (improvement / orig_acc * 100) if orig_acc > 0 else 0.0
        
        improvements[int(label)] = {
            'original': orig_acc,
            'learned': learned_acc,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
        
        class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else f"Class {int(label)}"
        print(f"{class_name:<15} {orig_acc:>11.4f}  {learned_acc:>11.4f}  {improvement:>+14.4f}  {improvement_pct:>+11.2f}%")
    
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1]['improvement'], reverse=True)
    top_improving = sorted_improvements[:5]
    
    print(f"\n{'='*60}")
    print(f"Top 5 Classes with Largest Improvement:")
    print(f"{'='*60}")
    for label, metrics in top_improving:
        class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else f"Class {int(label)}"
        print(f"{class_name}: {metrics['original']:.4f} → {metrics['learned']:.4f} "
              f"(+{metrics['improvement']:.4f}, {metrics['improvement_pct']:+.2f}%)")
    
    confusion_diff = learned_confusion - orig_confusion
    
    print(f"\n{'='*60}")
    print(f"Confusion Matrix Analysis:")
    print(f"{'='*60}")
    
    orig_misclass = np.sum(orig_confusion) - np.trace(orig_confusion)
    learned_misclass = np.sum(learned_confusion) - np.trace(learned_confusion)
    misclass_reduction = orig_misclass - learned_misclass
    misclass_reduction_pct = (misclass_reduction / orig_misclass * 100) if orig_misclass > 0 else 0.0
    
    print(f"Total Misclassifications: Original={orig_misclass}, Learned={learned_misclass}, "
          f"Reduction={misclass_reduction} ({misclass_reduction_pct:+.2f}%)")
    
    if len(unique_labels) <= 20:
        print(f"\nTop 10 Most Improved Confusion Pairs (fewer mistakes):")
        confusion_improvements = []
        for i, true_label in enumerate(unique_labels):
            for j, pred_label in enumerate(unique_labels):
                if i != j:  # Off-diagonal only
                    improvement = -confusion_diff[i, j]  # Negative = improvement (fewer mistakes)
                    if improvement > 0:
                        true_name = class_names[int(true_label)] if class_names and int(true_label) < len(class_names) else f"Class {int(true_label)}"
                        pred_name = class_names[int(pred_label)] if class_names and int(pred_label) < len(class_names) else f"Class {int(pred_label)}"
                        confusion_improvements.append((true_name, pred_name, improvement, orig_confusion[i, j], learned_confusion[i, j]))
        
        confusion_improvements.sort(key=lambda x: x[2], reverse=True)
        for true_name, pred_name, improvement, orig_count, learned_count in confusion_improvements[:10]:
            print(f"  {true_name} → {pred_name}: {orig_count} → {learned_count} (reduced by {int(improvement)})")
    
    results = {
        'original': {
            'class_accuracies': orig_class_acc,
            'confusion_matrix': orig_confusion,
            'predictions': orig_predictions
        },
        'learned': {
            'class_accuracies': learned_class_acc,
            'confusion_matrix': learned_confusion,
            'predictions': learned_predictions
        },
        'improvements': improvements,
        'confusion_difference': confusion_diff,
        'misclass_reduction': misclass_reduction,
        'misclass_reduction_pct': misclass_reduction_pct
    }
    
    return results


def local_pca_metric(neighborhood_X, min_neighbors=2):
    if neighborhood_X.shape[0] < min_neighbors:
        return 0.0, 0.0
    
    if neighborhood_X.shape[0] == 1:
        return 0.0, 0.0
    
    pca = PCA()
    pca.fit(neighborhood_X)
    eigenvalues = pca.explained_variance_ratio_
    
    if len(eigenvalues) > 1 and eigenvalues[1] > 0:
        linearity_score = eigenvalues[0] / eigenvalues[1]
        variance_in_pc1 = eigenvalues[0]
    else:
        linearity_score = 0.0
        variance_in_pc1 = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
    
    return linearity_score, variance_in_pc1



def load_raw_data(directory=None):
    if directory is None:
        directory = global_dir
    raw_X = np.load(os.path.join(directory, "raw_X.npy"))
    return raw_X


def load_z_proposed(directory=None):
    if directory is None:
        directory = global_dir
    z_proposed = np.load(os.path.join(directory, "Z_umap_proposed.npy"))
    return z_proposed

def load_z_original(directory=None):
    if directory is None:
        directory = global_dir
    z_original = np.load(os.path.join(directory, "Z_umap_original.npy"))
    return z_original

def run_knn_on_z(z_proposed, n_neighbors=15, metric="cosine"):
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
    knn.fit(z_proposed)
    knn_indices = knn.kneighbors(z_proposed, return_distance=False)
    if knn_indices[0, 0] != 0:
        knn_indices = np.column_stack([np.arange(len(z_proposed)), knn_indices])
    return knn_indices


def perform_knn_classification_accuracy_2d(directory=None, method='loo', k_values=[14], 
                                          use_weighted_vote=False, test_size=0.2, n_folds=5, 
                                          random_state=42, report_classwise=True):
    if directory is None:
        directory = global_dir
    
    if method == 'train_test':
        if use_weighted_vote:
            use_weighted_vote = False
    
    label_y = load_label_y(directory)
    
    try:
        z_proposed = load_z_proposed(directory)
        z_original = load_z_original(directory)
        print(f" Loaded 2D embeddings: Proposed shape={z_proposed.shape}, Original shape={z_original.shape}")
    except FileNotFoundError as e:
        print(f"  Could not load 2D embeddings: {e}")
        return {}
    
    if z_proposed.shape[1] > 2:
        print(f"Proposed embedding has {z_proposed.shape[1]} dimensions, using first 2")
        z_proposed = z_proposed[:, :2]
    if z_original.shape[1] > 2:
        print(f"Original embedding has {z_original.shape[1]} dimensions, using first 2")
        z_original = z_original[:, :2]
    
    n_neighbors = max(k_values) if k_values else 14
    print(f"\nComputing KNN graphs on 2D embeddings (k={n_neighbors}, metric='euclidean')")
    
    print("  Computing KNN on original UMAP 2D layout...")
    knn_original_2d = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    knn_original_2d.fit(z_original)
    original_2d_knn_indices = knn_original_2d.kneighbors(z_original, return_distance=False)
    if original_2d_knn_indices[0, 0] != 0:
        original_2d_knn_indices = np.column_stack([np.arange(len(z_original)), original_2d_knn_indices])
    
    print("  Computing KNN on proposed 2D layout...")
    knn_proposed_2d = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    knn_proposed_2d.fit(z_proposed)
    proposed_2d_knn_indices = knn_proposed_2d.kneighbors(z_proposed, return_distance=False)
    if proposed_2d_knn_indices[0, 0] != 0:
        proposed_2d_knn_indices = np.column_stack([np.arange(len(z_proposed)), proposed_2d_knn_indices])
    
    edge_weights = None
    if use_weighted_vote:
        try:
            from scipy.sparse import load_npz
            fuzzy_set_path = os.path.join(directory, "fuzzy_set.npz")
            if os.path.exists(fuzzy_set_path):
                fuzzy_set = load_npz(fuzzy_set_path)
                edge_weights = fuzzy_set.tocsr()
                pass
            else:
                use_weighted_vote = False
        except Exception as e:
            use_weighted_vote = False
    
    results = {}
    
    # Evaluate original UMAP 2D KNN graph
    print(f"\n{'='*60}")
    print(f"Evaluating Original UMAP 2D KNN Graph (method={method})")
    print(f"{'='*60}")
    original_2d_results = evaluate_knn_classification(
        knn_indices=original_2d_knn_indices,
        labels=label_y,
        method=method,
        k_values=k_values,
        use_weighted_vote=False,  # Original doesn't have learned weights
        edge_weights=None,
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state
    )
    results['original_2d'] = original_2d_results
    
    # Evaluate proposed 2D KNN graph
    print(f"\n{'='*60}")
    print(f"Evaluating Proposed UMAP 2D KNN Graph (method={method})")
    print(f"{'='*60}")
    proposed_2d_results = evaluate_knn_classification(
        knn_indices=proposed_2d_knn_indices,
        labels=label_y,
        method=method,
        k_values=k_values,
        use_weighted_vote=use_weighted_vote,
        edge_weights=edge_weights,
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state
    )
    results['proposed_2d'] = proposed_2d_results
    
    for k in k_values:
        orig_2d_acc = original_2d_results.get(f'accuracy_k{k}', 0.0)
        proposed_2d_acc = proposed_2d_results.get(f'accuracy_k{k}', 0.0)
        improvement = proposed_2d_acc - orig_2d_acc
        improvement_pct = (improvement / orig_2d_acc * 100) if orig_2d_acc > 0 else 0.0
        
        orig_2d_skipped = original_2d_results.get(f'skipped_points_k{k}', 0)
        proposed_2d_skipped = proposed_2d_results.get(f'skipped_points_k{k}', 0)
        orig_2d_skipped_pct = original_2d_results.get(f'skipped_pct_k{k}', 0.0)
        proposed_2d_skipped_pct = proposed_2d_results.get(f'skipped_pct_k{k}', 0.0)
        
        print(f"k={k:2d}: Original-2D={orig_2d_acc:.4f}, Proposed-2D={proposed_2d_acc:.4f}, "
              f"Improvement={improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if orig_2d_skipped > 0 or proposed_2d_skipped > 0:
            print(f"      Skipped: Original-2D={orig_2d_skipped} ({orig_2d_skipped_pct:.2f}%), "
                  f"Proposed-2D={proposed_2d_skipped} ({proposed_2d_skipped_pct:.2f}%)")
    
    # Class-wise 
    if method == 'loo' and report_classwise:
        class_names = None
        n_classes = len(np.unique(label_y))
        if n_classes == 10:
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        for k in k_values:
            if k <= original_2d_knn_indices.shape[1] - 1 and k <= proposed_2d_knn_indices.shape[1] - 1:
                classwise_results = report_classwise_comparison(
                    original_knn_indices=original_2d_knn_indices,
                    learned_knn_indices=proposed_2d_knn_indices,
                    labels=label_y,
                    k=k,
                    original_edge_weights=None,
                    learned_edge_weights=edge_weights,
                    use_weighted_vote=use_weighted_vote,
                    class_names=class_names
                )
                if 'classwise_2d' not in results:
                    results['classwise_2d'] = {}
                results['classwise_2d'][f'k{k}'] = classwise_results
    
    # Skipped points 
    if method == 'loo':
        class_names = None
        n_classes = len(np.unique(label_y))
        if n_classes == 10:
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        for k in k_values:
            if k <= original_2d_knn_indices.shape[1] - 1 and k <= proposed_2d_knn_indices.shape[1] - 1:
                original_2d_skipped_analysis = analyze_skipped_points(
                    knn_indices=original_2d_knn_indices,
                    labels=label_y,
                    k=k,
                    use_weighted_vote=False,
                    edge_weights=None,
                    z_2d=z_original,
                    class_names=class_names
                )
                
                proposed_2d_skipped_analysis = analyze_skipped_points(
                    knn_indices=proposed_2d_knn_indices,
                    labels=label_y,
                    k=k,
                    use_weighted_vote=use_weighted_vote,
                    edge_weights=edge_weights,
                    z_2d=z_proposed,
                    class_names=class_names
                )
                
                report_skipped_points_analysis(
                    original_analysis=original_2d_skipped_analysis,
                    learned_analysis=proposed_2d_skipped_analysis,
                    labels=label_y,
                    class_names=class_names
                )
                
                if 'skipped_analysis_2d' not in results:
                    results['skipped_analysis_2d'] = {}
                results['skipped_analysis_2d'][f'k{k}'] = {
                    'original_2d': original_2d_skipped_analysis,
                    'proposed_2d': proposed_2d_skipped_analysis
                }
    
    return results



def get_local_pca_metric(directory=None, n_neighbors=15, batch_size=1000, return_variance_pc1=False, use_embedding="proposed"):
    raw_X = load_raw_data(directory)
    
    if use_embedding == "proposed":
        z_embedding = load_z_proposed(directory)
        embedding_name = "proposed"
    elif use_embedding == "original":
        z_embedding = load_z_original(directory)
        embedding_name = "original"
    else:
        raise ValueError(f"use_embedding must be 'proposed' or 'original', got '{use_embedding}'")
    
    print(f"Using {embedding_name} embedding for KNN computation...")
    knn_indices = run_knn_on_z(z_embedding, n_neighbors=n_neighbors)
    
    n_points = raw_X.shape[0]
    linearity_scores = np.zeros(n_points)
    variance_pc1_scores = np.zeros(n_points)
    
    print(f"Computing local PCA metric for {n_points} points...")
    
    for i in tqdm(range(0, n_points, batch_size), desc="Processing neighborhoods"):
        batch_end = min(i + batch_size, n_points)
        batch_indices = np.arange(i, batch_end)
        
        for point_idx in batch_indices:
            neighbor_indices = knn_indices[point_idx]
            neighborhood_X = raw_X[neighbor_indices]
            
            linearity_score, variance_in_pc1 = local_pca_metric(neighborhood_X)
            linearity_scores[point_idx] = linearity_score
            variance_pc1_scores[point_idx] = variance_in_pc1
    
    if return_variance_pc1:
        return linearity_scores, variance_pc1_scores
    else:
        return linearity_scores


def compute_pr_and_r_from_knn(data, knn_indices, min_neighbors=2, batch_size=100, device=None, use_torch=True):
    if device is None:
        device = torch.device("cpu")
    
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        data_tensor = data
    
    if isinstance(knn_indices, np.ndarray):
        knn_indices = torch.from_numpy(knn_indices).long()
    
    data_tensor = data_tensor.to(device)
    knn_indices = knn_indices.to(device)
    
    n_points, C = data_tensor.shape
    pr_array = torch.zeros(n_points, device=device)
    r_array = torch.zeros(n_points, device=device)
    
    for batch_start in tqdm(range(0, n_points, batch_size), desc="Computing PR and R"):
        batch_end = min(batch_start + batch_size, n_points)
        batch_indices = list(range(batch_start, batch_end))
        
        batch_covs = []
        valid_mask = []
        
        for point_idx in batch_indices:
            neighbor_indices = knn_indices[point_idx]
            
            if len(neighbor_indices) >= min_neighbors:
                point_data = data_tensor[point_idx:point_idx+1, :]
                neighbor_data = data_tensor[neighbor_indices, :]
                local_manifold = torch.cat([point_data, neighbor_data], dim=0)
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
    
    return pr_array.cpu().numpy(), r_array.cpu().numpy()


def get_pr_and_r_metric(directory=None, n_neighbors=15, batch_size=100, 
                       use_embedding="proposed", compute_space="raw_X", 
                       knn_source="embedding", plot_embedding=None, device=None):
    raw_X = load_raw_data(directory)
    
    if plot_embedding is None:
        plot_embedding = use_embedding
    
    if knn_source == "raw_X":
        print(f"Using raw_X for KNN computation...")
        knn_indices = run_knn_on_z(raw_X, n_neighbors=n_neighbors, metric="cosine")
    else:
        if use_embedding == "proposed":
            z_embedding = load_z_proposed(directory)
            embedding_name = "proposed"
        elif use_embedding == "original":
            z_embedding = load_z_original(directory)
            embedding_name = "original"
        else:
            raise ValueError(f"use_embedding must be 'proposed' or 'original', got '{use_embedding}'")
        
        print(f"Using {embedding_name} embedding for KNN computation...")
        knn_indices = run_knn_on_z(z_embedding, n_neighbors=n_neighbors)
    
    if compute_space not in ["raw_X", "embedding"]:
        raise ValueError(f"compute_space must be 'raw_X' or 'embedding', got '{compute_space}'")
    
    if compute_space == "raw_X":
        data_for_computation = raw_X
        space_name = "raw_X"
    else:
        if knn_source == "raw_X":
            embedding_for_compute = plot_embedding
        else:
            embedding_for_compute = use_embedding
        
        if embedding_for_compute == "proposed":
            z_embedding = load_z_proposed(directory)
        else:
            z_embedding = load_z_original(directory)
        data_for_computation = z_embedding
        space_name = "embedding"
    
    print(f"Computing PR and R in {space_name} space...")
    pr_array, r_array = compute_pr_and_r_from_knn(
        data_for_computation, knn_indices, 
        min_neighbors=2, batch_size=batch_size, device=device
    )
    
    if plot_embedding == "proposed":
        z_plot = load_z_proposed(directory)
    else:
        z_plot = load_z_original(directory)
    
    if z_plot.shape[1] > 2:
        z_plot = z_plot[:, :2]
    
    return pr_array, r_array, z_plot



def plot_local_pca_metrics(z_embedding, linearity_scores, variance_pc1_scores, 
                           directory=None, save_path=None, figsize=(12, 5), 
                           point_size=1.0, alpha=0.6):
    if z_embedding.shape[1] != 2:
        raise ValueError(f"z_embedding must be 2D (got shape {z_embedding.shape})")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    scatter1 = axes[0].scatter(
        z_embedding[:, 0], 
        z_embedding[:, 1], 
        c=linearity_scores, 
        cmap='viridis',
        s=point_size,
        alpha=alpha,
        edgecolors='none'
    )
    axes[0].set_title('Local PCA Linearity Score', fontsize=12)
    axes[0].set_xlabel('UMAP Dimension 1', fontsize=10)
    axes[0].set_ylabel('UMAP Dimension 2', fontsize=10)
    axes[0].set_aspect('equal', adjustable='box')
    plt.colorbar(scatter1, ax=axes[0], label='Linearity Score')
    
    scatter2 = axes[1].scatter(
        z_embedding[:, 0], 
        z_embedding[:, 1], 
        c=variance_pc1_scores, 
        cmap='plasma',
        s=point_size,
        alpha=alpha,
        edgecolors='none'
    )
    axes[1].set_title('Variance in First Principal Component', fontsize=12)
    axes[1].set_xlabel('UMAP Dimension 1', fontsize=10)
    axes[1].set_ylabel('UMAP Dimension 2', fontsize=10)
    axes[1].set_aspect('equal', adjustable='box')
    plt.colorbar(scatter2, ax=axes[1], label='Variance in PC1')
    
    plt.tight_layout()
    
    if save_path is None and directory is not None:
        save_path = os.path.join(directory, "local_pca_metrics_plot.png")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pr_and_r_metrics(z_embedding, pr_array, r_array, 
                          directory=None, save_path=None, figsize=(12, 5), 
                          point_size=1.0, alpha=0.6):
    if z_embedding.shape[1] != 2:
        raise ValueError(f"z_embedding must be 2D (got shape {z_embedding.shape})")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    scatter1 = axes[0].scatter(
        z_embedding[:, 0], 
        z_embedding[:, 1], 
        c=pr_array, 
        cmap='coolwarm',
        s=point_size,
        alpha=alpha,
        edgecolors='none'
    )
    axes[0].set_title('PR Metric', fontsize=12)
    axes[0].set_xlabel('UMAP Dimension 1', fontsize=10)
    axes[0].set_ylabel('UMAP Dimension 2', fontsize=10)
    axes[0].set_aspect('equal', adjustable='box')
    plt.colorbar(scatter1, ax=axes[0], label='PR Value')
    
    scatter2 = axes[1].scatter(
        z_embedding[:, 0], 
        z_embedding[:, 1], 
        c=r_array, 
        cmap='inferno',
        s=point_size,
        alpha=alpha,
        edgecolors='none'
    )
    axes[1].set_title('R Metric', fontsize=12)
    axes[1].set_xlabel('UMAP Dimension 1', fontsize=10)
    axes[1].set_ylabel('UMAP Dimension 2', fontsize=10)
    axes[1].set_aspect('equal', adjustable='box')
    plt.colorbar(scatter2, ax=axes[1], label='R Value')
    
    plt.tight_layout()
    
    if save_path is None and directory is not None:
        save_path = os.path.join(directory, "pr_r_metrics_plot.png")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def perform_local_metric(use_embedding = "proposed", compute_metric = "both" ):
    if compute_metric in ["pca", "both"]:
        if use_embedding == "proposed":
            z_plot = load_z_proposed()
        else:
            z_plot = load_z_original()
        
        if z_plot.shape[1] > 2:
            z_plot = z_plot[:, :2]
        
        linearity_scores, variance_pc1_scores = get_local_pca_metric(return_variance_pc1=True, use_embedding=use_embedding)
        
        save_filename = f"local_pca_metrics_plot_{use_embedding}.png"
        plot_local_pca_metrics(z_plot, linearity_scores, variance_pc1_scores, 
                              directory=global_dir, save_path=os.path.join(global_dir, save_filename))
    
    if compute_metric in ["pr_r", "both"]:
        compute_space = "raw_X"
        knn_source = "raw_X"
        embeddings_to_process = ["original", "proposed"]
        
        for plot_embedding_type in embeddings_to_process:
            pr_array, r_array, z_plot = get_pr_and_r_metric(
                use_embedding="proposed",
                compute_space=compute_space,
                knn_source=knn_source,
                plot_embedding=plot_embedding_type
            )
            
            save_filename = f"pr_r_metrics_plot_knn_{knn_source}_space_{compute_space}_plot_{plot_embedding_type}.png"
            plot_pr_and_r_metrics(z_plot, pr_array, r_array, 
                                 directory=global_dir, save_path=os.path.join(global_dir, save_filename))



def batch_evaluate_results_directories(results_dir=None, k_values=[10], output_json_path=None):
    import json
    from pathlib import Path
    
    if results_dir is None:
        script_dir = Path(__file__).parent.parent.parent
        results_dir = script_dir / "src_v2" / "results"
    else:
        results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return {}
    
    if output_json_path is None:
        output_json_path = results_dir / "results_batch_evaluation.json"
    else:
        output_json_path = Path(output_json_path)
    
    all_results = {
        "description": "Batch evaluation of all result directories",
        "computation_method": "Using 2D embeddings (Z_umap_original, Z_umap_proposed) with kNN classification",
        "k_values": k_values,
        "total_directories_processed": 0,
        "total_directories_skipped": 0,
        "results": []
    }
    
    subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    
    for subdir in subdirs:
        z_original_path = subdir / "Z_umap_original.npy"
        z_proposed_path = subdir / "Z_umap_proposed.npy"
        y_path = subdir / "y.npy"
        
        if not (z_original_path.exists() and z_proposed_path.exists() and y_path.exists()):
            all_results["total_directories_skipped"] += 1
            continue
        
        try:
            Z_umap_original = np.load(z_original_path)
            Z_umap_proposed = np.load(z_proposed_path)
            y = np.load(y_path)
            
            if Z_umap_original.shape[1] > 2:
                Z_umap_original = Z_umap_original[:, :2]
            if Z_umap_proposed.shape[1] > 2:
                Z_umap_proposed = Z_umap_proposed[:, :2]
            
            if len(y) != len(Z_umap_original) or len(y) != len(Z_umap_proposed):
                all_results["total_directories_skipped"] += 1
                continue
            
            n_neighbors = max(k_values) if k_values else 10
            
            knn_original_2d = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
            knn_original_2d.fit(Z_umap_original)
            original_2d_knn_indices = knn_original_2d.kneighbors(Z_umap_original, return_distance=False)
            if original_2d_knn_indices[0, 0] != 0:
                original_2d_knn_indices = np.column_stack([np.arange(len(Z_umap_original)), original_2d_knn_indices])
            
            knn_proposed_2d = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
            knn_proposed_2d.fit(Z_umap_proposed)
            proposed_2d_knn_indices = knn_proposed_2d.kneighbors(Z_umap_proposed, return_distance=False)
            if proposed_2d_knn_indices[0, 0] != 0:
                proposed_2d_knn_indices = np.column_stack([np.arange(len(Z_umap_proposed)), proposed_2d_knn_indices])
            
            result_entry = {
                "directory_name": subdir.name,
                "directory_path": str(subdir),
                "n_samples": len(y),
                "n_classes": len(np.unique(y)),
                "class_labels": sorted(np.unique(y).tolist()),
                "knn_accuracy": {},
                "per_class_accuracy": {},
                "confusion_matrices": {}
            }
            
            for k in k_values:
                if k > original_2d_knn_indices.shape[1] - 1 or k > proposed_2d_knn_indices.shape[1] - 1:
                    continue
                
                k_key = f"k{k}"
                
                original_2d_results = evaluate_knn_classification(
                    knn_indices=original_2d_knn_indices,
                    labels=y,
                    method='loo',
                    k_values=[k],
                    use_weighted_vote=False,
                    edge_weights=None,
                    random_state=42
                )
                
                proposed_2d_results = evaluate_knn_classification(
                    knn_indices=proposed_2d_knn_indices,
                    labels=y,
                    method='loo',
                    k_values=[k],
                    use_weighted_vote=False,
                    edge_weights=None,
                    random_state=42
                )
                
                orig_acc = original_2d_results.get(f'accuracy_k{k}', None)
                prop_acc = proposed_2d_results.get(f'accuracy_k{k}', None)
                
                result_entry["knn_accuracy"][k_key] = {
                    "original_umap": float(orig_acc) if orig_acc is not None else None,
                    "proposed_maple": float(prop_acc) if prop_acc is not None else None,
                    "improvement": float(prop_acc - orig_acc) if (orig_acc is not None and prop_acc is not None) else None
                }
                
                orig_predictions, orig_class_acc, orig_confusion = compute_classwise_accuracy(
                    knn_indices=original_2d_knn_indices,
                    labels=y,
                    k=k,
                    use_weighted_vote=False,
                    edge_weights=None
                )
                
                prop_predictions, prop_class_acc, prop_confusion = compute_classwise_accuracy(
                    knn_indices=proposed_2d_knn_indices,
                    labels=y,
                    k=k,
                    use_weighted_vote=False,
                    edge_weights=None
                )
                
                result_entry["per_class_accuracy"][k_key] = {
                    "original_umap": {str(int(cls)): float(acc) for cls, acc in orig_class_acc.items()},
                    "proposed_maple": {str(int(cls)): float(acc) for cls, acc in prop_class_acc.items()},
                    "improvement": {str(int(cls)): float(prop_class_acc.get(cls, 0) - orig_class_acc.get(cls, 0)) 
                                   for cls in set(list(orig_class_acc.keys()) + list(prop_class_acc.keys()))}
                }
                
                result_entry["confusion_matrices"][k_key] = {
                    "original_umap": orig_confusion.tolist(),
                    "proposed_maple": prop_confusion.tolist()
                }
            
            all_results["results"].append(result_entry)
            all_results["total_directories_processed"] += 1
            
        except Exception as e:
            all_results["total_directories_skipped"] += 1
    
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def select_best_per_dataset(input_json_path=None, output_json_path=None, k_value=10):
    import json
    from pathlib import Path
    import re
    
    if input_json_path is None:
        script_dir = Path(__file__).parent.parent.parent
        results_dir = script_dir / "src_v2" / "results"
        input_json_path = results_dir / "results_batch_evaluation.json"
    else:
        input_json_path = Path(input_json_path)
        results_dir = input_json_path.parent
    
    if not input_json_path.exists():
        print(f"Error: Input JSON file not found at {input_json_path}")
        return {}
    
    if output_json_path is None:
        output_json_path = results_dir / "results_batch_selected.json"
    else:
        output_json_path = Path(output_json_path)
    
    print(f"Loading batch evaluation results from: {input_json_path}")
    with open(input_json_path, 'r') as f:
        all_results = json.load(f)
    
    k_key = f"k{k_value}"
    
    dataset_groups = {}
    
    for result in all_results.get("results", []):
        directory_name = result.get("directory_name", "")
        
        if not directory_name:
            continue
        
        knn_accuracy = result.get("knn_accuracy", {}).get(k_key, {})
        improvement = knn_accuracy.get("improvement")
        
        if improvement is None:
            continue
        
        dataset_name = None
        parts = directory_name.split("_")
        
        if len(parts) >= 2:
            dataset_parts = []
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) >= 4:
                    break
                dataset_parts.append(part)
            
            if dataset_parts:
                dataset_name = "_".join(dataset_parts)
            else:
                dataset_name = parts[0] if parts else directory_name
        else:
            dataset_name = directory_name
        
        if dataset_name not in dataset_groups:
            dataset_groups[dataset_name] = []
        
        dataset_groups[dataset_name].append((improvement, result))
    
    selected_results = {
        "description": "Selected best result per dataset (maximum improvement)",
        "source_file": str(input_json_path),
        "k_value_used": k_value,
        "total_datasets": len(dataset_groups),
        "total_directories_selected": 0,
        "results": []
    }
    
    for dataset_name, entries in sorted(dataset_groups.items()):
        if not entries:
            continue
        
        max_improvement_entry = max(entries, key=lambda x: x[0])
        max_improvement, best_result = max_improvement_entry
        
        selected_results["results"].append(best_result)
        selected_results["total_directories_selected"] += 1
    
    with open(output_json_path, 'w') as f:
        json.dump(selected_results, f, indent=2)
    
    return selected_results


def run_umap_and_evaluate(dataset_name="fashion_mnist_full", dataset_size=40000, 
                          n_neighbors=15, k_values=[5, 10, 15, 30], 
                          save_results=False, output_dir=None):
    import numpy as np
    import os
    from data.toy_data import loading_toy_data
    from utils.umap_utils import umap_original
    from eval_metrics.zadu.measures.neighborhood_hit import measure as neighborhood_hit
    from sklearn.neighbors import NearestNeighbors
    
    raw_X, y = loading_toy_data(dataset_name, number=dataset_size)
    
    if isinstance(raw_X, np.ndarray):
        raw_X_np = raw_X
    else:
        raw_X_np = raw_X.numpy() if hasattr(raw_X, 'numpy') else np.array(raw_X)
    
    if isinstance(y, np.ndarray):
        y_np = y
    else:
        y_np = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    
    Z_umap = umap_original(raw_X_np, n_neighbors=n_neighbors, verbose=True)
    nh_result = neighborhood_hit(Z_umap, y_np, k=n_neighbors, return_local=False)
    nh_score = nh_result["neighborhood_hit"]
    
    max_k = max(k_values) if k_values else 15
    knn_2d = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", n_jobs=-1)
    knn_2d.fit(Z_umap)
    umap_2d_knn_indices = knn_2d.kneighbors(Z_umap, return_distance=False)
    if umap_2d_knn_indices[0, 0] != 0:
        umap_2d_knn_indices = np.column_stack([np.arange(len(Z_umap)), umap_2d_knn_indices])
    
    knn_results = {}
    for k in k_values:
        if k > umap_2d_knn_indices.shape[1] - 1:
            continue
        
        k_results = evaluate_knn_classification(
            knn_indices=umap_2d_knn_indices,
            labels=y_np,
            method='loo',
            k_values=[k],
            use_weighted_vote=False,
            edge_weights=None,
            random_state=42
        )
        knn_results[f'accuracy_k{k}'] = k_results.get(f'accuracy_k{k}', None)
        knn_results[f'skipped_points_k{k}'] = k_results.get(f'skipped_points_k{k}', 0)
    
    if save_results:
        if output_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"umap_results_{dataset_name}_{dataset_size}_{n_neighbors}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "Z_umap_original.npy"), Z_umap)
        np.save(os.path.join(output_dir, "raw_X.npy"), raw_X_np)
        np.save(os.path.join(output_dir, "y.npy"), y_np)
    
    results = {
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "n_neighbors": n_neighbors,
        "neighborhood_hit": {
            "k": n_neighbors,
            "score": float(nh_score)
        },
        "knn_classification_accuracy": knn_results
    }
    
    return results


if __name__ == "__main__":
    results = run_umap_and_evaluate(
        dataset_name="fashion_mnist_full",
        dataset_size=40000,
        n_neighbors=15,
        k_values=[5, 10, 15, 30],
        save_results=False
    )
    print("\nFinal Results:", results)
    
