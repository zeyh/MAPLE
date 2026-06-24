import numpy as np
from umap.umap_ import make_epochs_per_sample, noisy_scale_coords, spectral_layout, find_ab_params # import from original umap repo
from utils.umap_utils import optimize_layout_generic

import os # TOFIX: Disable all parallel processing to avoid memory leak 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
os.environ['LAPACK_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def ce_umap_original(original_X, fuzzy_set):
    #wrapper function that delegates to ce_umap_robust for memory leak 
    return ce_umap_robust(original_X, fuzzy_set, max_spectral_size=40001, max_spectral_dim=5000)


def ce_umap_robust(original_X, fuzzy_set, max_spectral_size=40001, max_spectral_dim=5000):
    """
    wrapper that switches initialization strategy based on data size 
    to prevent memory leaks in spectral_layout.
    For datasets larger than max_spectral_size samples OR with dimension > max_spectral_dim,
    uses random initialization instead of spectral initialization to avoid O(N^3) memory issues.
    """
    import umap.distances as dist
    import gc
    import os as os_module
    
    try:
        import psutil
        process = psutil.Process(os_module.getpid())
        monitor_memory = True
    except ImportError:
        monitor_memory = False
    
    random_state_int = 42
    random_state = np.random.RandomState(random_state_int)
    np.random.seed(random_state_int)

    output_embedding_dim = 2
    n_epochs = 500
    min_dist = 0.1
    a, b = find_ab_params(spread=1.0, min_dist=min_dist)
    a = np.float32(a)
    b = np.float32(b)

    n_samples = original_X.shape[0]
    n_dim = original_X.shape[1]
    print(f"🔍 Input Size: {n_samples}, Dimension: {n_dim}")

    use_random_init = (n_samples > max_spectral_size) or (n_dim > max_spectral_dim)
    
    if use_random_init:
        reason = []
        if n_samples > max_spectral_size:
            reason.append(f"N > {max_spectral_size}")
        if n_dim > max_spectral_dim:
            reason.append(f"D > {max_spectral_dim}")
        print(f"🚀 Large data detected ({', '.join(reason)}). Using Random Initialization to avoid Memory Leak.")
        if monitor_memory:
            print(f"   Memory before initialization: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        initial_embedding = random_state.uniform(
            low=-10.0,
            high=10.0,
            size=(n_samples, output_embedding_dim)
        ).astype(np.float32, order="C")
        
        initial_embedding = noisy_scale_coords(
            initial_embedding, random_state=random_state, max_coord=10, noise=0.0001
        )
    else:
        print(f"📉 Small data detected. Using Spectral Initialization.")
        if monitor_memory:
            print(f"   Memory before spectral_layout: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        try:
            initial_embedding = spectral_layout(
                data=original_X + np.random.normal(scale=1e-2, size=original_X.shape),
                graph=fuzzy_set,
                dim=output_embedding_dim,
                random_state=random_state,
                metric="euclidean",
                metric_kwds={},
                tol=1e-4,
                maxiter=100
            )
            
            if monitor_memory:
                print(f"   Memory after spectral_layout: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            
            initial_embedding = (
                10.0
                * (initial_embedding - np.min(initial_embedding, 0))
                / (np.max(initial_embedding, 0) - np.min(initial_embedding, 0))
            ).astype(np.float32, order="C")
            
            initial_embedding = noisy_scale_coords(
                initial_embedding, random_state=random_state, max_coord=10, noise=0.0001
            )
            
        except Exception as e:
            print(f"⚠️ Spectral layout failed ({e}). Falling back to random.")
            initial_embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(n_samples, output_embedding_dim)
            ).astype(np.float32, order="C")
            
            initial_embedding = noisy_scale_coords(
                initial_embedding, random_state=random_state, max_coord=10, noise=0.0001
            )

    gc.collect()

    head = fuzzy_set.row.astype(np.int32)
    tail = fuzzy_set.col.astype(np.int32)
    scaled_epochs = 1.0
    weight = fuzzy_set.data
    epochs_per_sample = make_epochs_per_sample(weight, n_epochs) * scaled_epochs
    epochs_per_sample = epochs_per_sample.astype(np.float32)

    INT32_MIN = np.iinfo(np.int32).min
    INT32_MAX = np.iinfo(np.int32).max
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    if monitor_memory:
        print(f"⚡ Starting Optimization (RAM: {process.memory_info().rss / 1024 / 1024:.1f} MB)...")
    print("start optimize layout -----")
    
    try:
        Z_umap = optimize_layout_generic(
            head_embedding=initial_embedding,
            tail_embedding=initial_embedding,
            head=head,
            tail=tail,
            n_epochs=n_epochs,
            n_vertices=n_samples,
            epochs_per_sample=epochs_per_sample,
            a=a,
            b=b,
            gamma=1.0,
            initial_alpha=1.0,
            negative_sample_rate=5,
            move_other=True,
            rng_state=rng_state,
            output_metric=dist.named_distances_with_gradients["euclidean"],
            output_metric_kwds=(),
            verbose=True,
        )
        if monitor_memory:
            print(f"🔍 After optimize_layout_generic: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        gc.collect()
    except Exception as e:
        print(f"❌ Error in optimize_layout_generic: {e}")
        raise

    return Z_umap


def debug_fuzzy_set_quality(fuzzy_set, name="fuzzy_set"):
    import numpy as np
    from scipy.sparse import csr_matrix
    
    print(f"\n🔍 Debugging {name}:")
    print(f"   Shape: {fuzzy_set.shape}")
    print(f"   Number of edges: {fuzzy_set.nnz}")
    print(f"   Sparsity: {1 - fuzzy_set.nnz / (fuzzy_set.shape[0] * fuzzy_set.shape[1]):.4f}")
    
    # Check for zero-degree nodes
    degrees = np.array(fuzzy_set.sum(axis=1)).flatten()
    zero_degree_nodes = np.where(degrees == 0)[0]
    print(f"   Zero-degree nodes: {len(zero_degree_nodes)}")
    
    if len(zero_degree_nodes) > 0:
        print(f"   ⚠️  Zero-degree node indices: {zero_degree_nodes[:10]}...")
    
    # Check weight statistics
    if fuzzy_set.data.size > 0:
        print(f"   Weight statistics:")
        print(f"     Min: {fuzzy_set.data.min():.6f}")
        print(f"     Max: {fuzzy_set.data.max():.6f}")
        print(f"     Mean: {fuzzy_set.data.mean():.6f}")
        print(f"     Std: {fuzzy_set.data.std():.6f}")
    
    # Check connectivity
    csr_fuzzy = csr_matrix(fuzzy_set)
    try:
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(csr_fuzzy, directed=False)
        print(f"   Connected components: {n_components}")
        if n_components > 1:
            print(f"   ⚠️  Graph is not connected! Component sizes: {np.bincount(labels)}")
    except Exception as e:
        print(f"   ⚠️  Could not check connectivity: {e}")
    
    try:
        # Compute Laplacian matrix
        D = csr_fuzzy.sum(axis=1).A1  # Degree matrix (diagonal)
        L = csr_fuzzy.copy()
        L.setdiag(-D)  # Laplacian = A - D
        
        # Check if Laplacian is positive semi-definite
        eigenvals = np.linalg.eigvalsh(L.toarray())
        print(f"   Laplacian eigenvalues:")
        print(f"     Min: {eigenvals.min():.6f}")
        print(f"     Max: {eigenvals.max():.6f}")
        print(f"     Number of zero eigenvalues: {np.sum(np.abs(eigenvals) < 1e-10)}")
        
        # Check eigengap
        sorted_eigenvals = np.sort(eigenvals)
        eigengap = sorted_eigenvals[1] - sorted_eigenvals[0]  # Second smallest - smallest
        print(f"   Eigengap: {eigengap:.6f}")
        
        if eigengap < 1e-6:
            print(f"   ⚠️  Eigengap too small! This will cause spectral initialization to fail.")
        
    except Exception as e:
        print(f"   ⚠️  Could not compute spectral properties: {e}")
    
    return fuzzy_set
