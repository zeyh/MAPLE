import numpy as np
import torch


def read_pbmc():
    import scanpy as sc
    import pandas as pd

    # Load PBMC data
    adata = sc.read_10x_mtx(
        "data/local_data/pbmc3k/filtered_gene_bc_matrices/hg19/",
        var_names="gene_symbols",
        cache=True
    )

    # X: dense matrix
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)  # shape [2700, 32738]

    # y: load k-means cluster labels
    df_labels = pd.read_csv("data/local_data/pbmc3k/analysis/kmeans/7_clusters/clusters.csv")
    # Now df_labels.columns = ['Barcode', 'Cluster']
    y = torch.tensor(df_labels['Cluster'].values, dtype=torch.long)  # shape: [2700]

    print(X.shape, y.shape)  # Should be (2700, 32738), (2700,)

    return X, y


def read_and_visualize_tsne_only():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load cluster labels
    df_labels = pd.read_csv("data/local_data/pbmc3k/analysis/kmeans/7_clusters/clusters.csv")
    print(f"Cluster labels shape: {df_labels.shape}")
    print(f"Cluster labels columns: {df_labels.columns.tolist()}")

    # Load t-SNE projection
    df_tsne = pd.read_csv("data/local_data/pbmc3k/analysis/tsne/projection.csv")
    print(f"t-SNE projection shape: {df_tsne.shape}")
    print(f"t-SNE projection columns: {df_tsne.columns.tolist()}")
    
    # Create a mapping from barcode to cluster for alignment
    barcode_to_cluster = dict(zip(df_labels['Barcode'], df_labels['Cluster']))
    
    # Align t-SNE coordinates with cluster labels
    tsne_coords = []
    aligned_labels = []
    aligned_barcodes = []
    
    for _, row in df_tsne.iterrows():
        barcode = row['Barcode']
        if barcode in barcode_to_cluster:
            tsne_coords.append([row['TSNE-1'], row['TSNE-2']])
            aligned_labels.append(barcode_to_cluster[barcode])
            aligned_barcodes.append(barcode)
    
    tsne_coords = np.array(tsne_coords)
    aligned_labels = np.array(aligned_labels)
    
    print(f"\nAligned data shapes:")
    print(f"t-SNE coords: {tsne_coords.shape}")
    print(f"Aligned labels: {aligned_labels.shape}")
    print(f"Number of matched cells: {len(aligned_barcodes)}")
    
    # Visualize t-SNE projection with cluster colors
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                         c=aligned_labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Projection of PBMC Data with K-means Clusters')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.grid(True, alpha=0.3)
    
    # Add cluster statistics
    unique_clusters, counts = np.unique(aligned_labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster}: {count} cells")
    
    plt.tight_layout()
    plt.show()
    
    return tsne_coords, aligned_labels, aligned_barcodes

    
if __name__ == "__main__":
    # read_pbmc()
    tsne_coords, aligned_labels, aligned_barcodes = read_and_visualize_tsne_only()

# https://mespadoto.github.io/proj-quant-eval/post/datasets/
def read_data(datasetName="bank", number=2000):
    # reaed the x.npy and y.npy from the data/datasetname/ directory
    x = np.load(f"data/local_data/{datasetName}/X.npy")
    y = np.load(f"data/local_data/{datasetName}/y.npy")
    return x, y


def get_x_toy():
    X_toy = [
        [1.2, 2.4, 0.3, 0.5],
        [1.5, 2.5, 0.1, 0.2],
        [0.8, 2.6, 1, 0.3],
        [4.0, 5.2, 1.3, 0.4],
        [4.2, 5.0, 1.1, 0.5],
        [3.8, 5.4, 1.4, 0.6],
        [7.2, 8.4, 2.1, 0.7],
        [7.5, 8.3, 2.0, 0.8],
        [7.1, 8.2, 2.2, 0.9],
        [6.9, 8.0, 2.3, 1.0]
    ]
    y_toy = [
        0, 0, 0,   # Points 0, 1, 2 - class 0
        1, 1, 1,   
        2, 2, 2, 2  
    ]
    return np.array(X_toy), np.array(y_toy)

def generate_high_dim_data(N=20, D=100):
    X = np.random.randn(N, D)  # Random Gaussian noise
    y = np.array([i % 2 for i in range(N)])  # Binary labels 
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def generate_low_dim_data(N=1000, D=3):
    # Create N points randomly in D dimensions
    X = np.random.randn(N, D)  # Gaussian noise
    y = np.array([i % 5 for i in range(N)])  # 5 classes for variety
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def generate_roll(N=1000):
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=N, noise=0.1)
    distances_squared = X[:, 0] ** 2 + X[:, 2] ** 2
    threshold = np.median(distances_squared)  
    y = distances_squared < threshold
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def generate_checkerboard_3d(N=1000, noise=0.05):
    X = np.random.rand(N, 3) * 4 - 2  # 3D range between -2 and 2
    y = (((X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int)) % 2)
    X += noise * np.random.randn(N, 3)  # Add noise
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def stack_datasets(X1, y1, X2, y2, shift=10):
    X2_shifted = X2.clone()
    X2_shifted[:, 0] += shift  

    max_label_y1 = y1.max().item()
    y2_adjusted = y2 + max_label_y1 + 1

    X_stacked = torch.cat((X1, X2_shifted), dim=0)
    y_stacked = torch.cat((y1, y2_adjusted), dim=0)

    return X_stacked, y_stacked


def get_wine_data():
    from sklearn.datasets import load_wine
    dataset = load_wine()
    def scaler(x): return x  # as is (no sclaign)
    # scaler = StandardScaler().fit_transform
    # scaler = MinMaxScaler().fit_transform

    X = dataset.data
    X = scaler(X)
    y = dataset.target
    y_color = y

    return X, y_color


def loading_toy_data(dataset_name, number=2000, pc_size=None, is_saving=False):
    if dataset_name == "digits":
        from sklearn.datasets import load_digits
        dataset = load_digits()
        X = dataset.data
        y = dataset.target
    elif dataset_name == "toy":
        X, y = get_x_toy()
    elif dataset_name == "wine":
        X_wine, y_wine = get_wine_data()
        X = X_wine
        y = y_wine
    elif dataset_name == "high_dim":
        X_high_dim, y_high_dim = generate_high_dim_data(N=20, D=100)
        X = X_high_dim
        y = y_high_dim
    elif dataset_name == "low_dim_stacked":
        X_checkerboard, y_checkerboard = generate_checkerboard_3d()
        X_roll, y_roll = generate_roll()
        X_low_dim, y_low_dim = stack_datasets(X_checkerboard, y_checkerboard, X_roll, y_roll)
        X = X_low_dim
        y = y_low_dim
    elif dataset_name == "checkerboard":
        X_checkerboard, y_checkerboard = generate_checkerboard_3d()
        X = X_checkerboard
        y = y_checkerboard
    elif dataset_name == "roll":
        X_roll, y_roll = generate_roll()
        X = X_roll
        y = y_roll
    elif dataset_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        X = load_breast_cancer().data
        y = load_breast_cancer().target
    elif dataset_name == "mnist_test":
        from torchvision.datasets import MNIST
        test_dataset = MNIST(
            root='./data',          
            train=False,            
            transform=None,         
            download=True           
        )
        X = test_dataset.data.float()  # [10000, 28, 28]
        X = X.reshape(X.shape[0], -1)  # [10000, 784]
        X = X / 255.0                  # normalize to 0-1
        y = test_dataset.targets       # [10000]
        sample_size = number
        X = X[:sample_size]
        y = y[:sample_size]

        
    elif dataset_name == "fashion_mnist_full":
        from torchvision.datasets import FashionMNIST
        import torch
        
        # Load both train and test sets
        train_dataset = FashionMNIST(root='./data', train=True, transform=None, download=True)
        test_dataset = FashionMNIST(root='./data', train=False, transform=None, download=True)
        
        # Combine the data
        X_train = train_dataset.data.float()  # [60000, 28, 28]
        X_test = test_dataset.data.float()    # [10000, 28, 28]
        X = torch.cat([X_train, X_test], dim=0)  # [70000, 28, 28]
        
        X = X.reshape(X.shape[0], -1)  # [70000, 784]
        X = X / 255.0                  # normalize to 0-1
        
        y_train = train_dataset.targets  # [60000]
        y_test = test_dataset.targets    # [10000]
        y = torch.cat([y_train, y_test], dim=0)  # [70000]
        
        if number != -1:
            sample_size = number
        else:
            sample_size = len(X)  # Now this is 70000
        X = X[:sample_size]
        y = y[:sample_size]
    elif dataset_name == "pbmc3k":
        X, y = read_pbmc()

    elif dataset_name == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups_vectorized
        import torch
        
        # Load the dataset
        print("Loading 20 newsgroups vectorized dataset...")
        newsgroups = fetch_20newsgroups_vectorized(subset='all', remove=('headers', 'footers', 'quotes'))
        
        # Convert to tensors
        X = torch.tensor(newsgroups.data.toarray(), dtype=torch.float32)  # Convert sparse matrix to dense
        y = torch.tensor(newsgroups.target, dtype=torch.long)
        
        print(f"20 Newsgroups Dataset:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Class names: {newsgroups.target_names}")
        
        # Apply sample size limit if specified
        if number is not None:
            sample_size = min(number, len(X))
            X = X[:sample_size]
            y = y[:sample_size]
            print(f"  Using {sample_size} samples (limited by number parameter)")


    elif dataset_name == "rcv1_multilabel":
        from sklearn.datasets import fetch_rcv1
        import torch
        import numpy as np
        
        # Load RCV1 dataset
        print("Loading RCV1 dataset (multi-label)...")
        rcv1 = fetch_rcv1(subset='all', download_if_missing=True, shuffle=False, random_state=42)
        
        # Sample 10K samples to avoid memory issues
        sample_size = 10000 if number is None else min(number, 10000)
        
        # Sample indices randomly
        np.random.seed(42)
        sample_indices = np.random.choice(rcv1.data.shape[0], size=sample_size, replace=False)
        
        # Extract sampled data
        X_sparse = rcv1.data[sample_indices]
        y_sparse = rcv1.target[sample_indices]
        
        # Convert to dense tensors (now much smaller)
        X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
        y_multi = torch.tensor(y_sparse.toarray(), dtype=torch.float32)
    
        # Convert multi-label to single-label (take first non-zero label)
        y = torch.zeros(len(X), dtype=torch.long)
        for i in range(len(X)):
            non_zero_indices = torch.nonzero(y_multi[i]).flatten()
            if len(non_zero_indices) > 0:
                y[i] = non_zero_indices[0]  # Take first label
            else:
                y[i] = 0  # Default to class 0 if no labels
        
        print(f"RCV1 Dataset (converted to single-label):")
        print(f"  Original dataset size: {rcv1.data.shape[0]} samples")
        print(f"  Sampled samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Class labels: {sorted(torch.unique(y).tolist())}")
        
        return X, y

    elif dataset_name == "STL_10":
        from torchvision.datasets import STL10
        import torch
        
        # Load train dataset (5000 samples)
        train_dataset = STL10(
            root='./data',
            split='train',  # 'train', 'test', or 'unlabeled'
            transform=None,
            download=True
        )
        
        # Load test dataset (8000 samples)
        test_dataset = STL10(
            root='./data',
            split='test',
            transform=None,
            download=True
        )

        import torch
        X_train = torch.tensor(train_dataset.data, dtype=torch.float32)  # [500, 3, 96, 96]
        X_train = X_train.reshape(X_train.shape[0], -1)  # [500, 3*96*96] = [500, 27648]
        X_train = X_train / 255.0  # normalize to 0-1
        y_train = torch.tensor(train_dataset.labels, dtype=torch.long)  # [500]
        
        X_test = torch.tensor(test_dataset.data, dtype=torch.float32)  # [8000, 3, 96, 96]
        X_test = X_test.reshape(X_test.shape[0], -1)  # [8000, 3*96*96] = [8000, 27648]
        X_test = X_test / 255.0  # normalize to 0-1
        y_test = torch.tensor(test_dataset.labels, dtype=torch.long)  # [8000]
        
        # Combine train and test data
        X = torch.cat([X_train, X_test], dim=0)  # [8500, 27648]
        y = torch.cat([y_train, y_test], dim=0)  # [8500]
        
            
        print(f"STL-10 Combined Dataset:")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        
        # Apply sample size limit
        if number is not None:
            sample_size = min(number, len(X))
            X = X[:sample_size]
            y = y[:sample_size]
            print(f"  Using {sample_size} samples (limited by number parameter)")
        
        return X, y
    
    elif dataset_name == "cifar10_full":
        from torchvision.datasets import CIFAR10
        import torch
        # Load both train and test sets
        train_dataset = CIFAR10(root='./data', train=True, transform=None, download=True)
        test_dataset = CIFAR10(root='./data', train=False, transform=None, download=True)
        
        # Combine the data
        X_train = torch.tensor(train_dataset.data, dtype=torch.float32)  # [50000, 32, 32, 3]
        X_test = torch.tensor(test_dataset.data, dtype=torch.float32)    # [10000, 32, 32, 3]
        X = torch.cat([X_train, X_test], dim=0)  # [60000, 32, 32, 3]
        
        # Convert from [N, H, W, C] to [N, C, H, W] format
        X = X.permute(0, 3, 1, 2)  # [60000, 3, 32, 32]
        X = X.reshape(X.shape[0], -1)  # [60000, 3*32*32] = [60000, 3072]
        X = X / 255.0                  # normalize to 0-1
        
        y_train = torch.tensor(train_dataset.targets, dtype=torch.long)  # [50000]
        y_test = torch.tensor(test_dataset.targets, dtype=torch.long)    # [10000]
        y = torch.cat([y_train, y_test], dim=0)  # [60000]
        
        if number != -1:
            sample_size = number
        else:
            sample_size = len(X)  # Now this is 60000
        X = X[:sample_size]
        y = y[:sample_size]
        
        print(f"CIFAR10 Full Dataset:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Image shape: (3, 32, 32)")
        
        return X, y
    
    elif dataset_name == "celegans_l2_neuron":
        import numpy as np
        import torch
        file_dir = "data/local_data/CelegansL2Cello_r/neuron"
        X = np.load(f"{file_dir}/X_neur.npy")
        y = np.load(f"{file_dir}/y_labels.npy")

        # to torch tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        print(f"C. elegans L2 Neuron Dataset:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Class labels: {sorted(torch.unique(y).tolist())}")

        return X, y

    elif dataset_name == "celegans_full":
        import numpy as np
        import torch
        file_dir = "data/local_data/CelegansL2Cello_r/full"
        X = np.load(f"{file_dir}/X_full.npy")
        y = np.load(f"{file_dir}/y_labels.npy")

        # to torch tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        print(f"C. elegans L2  Dataset:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Class labels: {sorted(torch.unique(y).tolist())}")

        return X, y
    
    elif dataset_name == "celegans_l2_glia":
        import numpy as np
        import torch
        file_dir = "data/local_data/CelegansL2Cello_r/glia"
        X = np.load(f"{file_dir}/X_glia.npy")
        y = np.load(f"{file_dir}/y_labels.npy")

        # to torch tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        print(f"C. elegans L2 Glia Dataset:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Class labels: {sorted(torch.unique(y).tolist())}")

        return X, y

    elif dataset_name == "celegans_processed":
        import pandas as pd
        import numpy as np
        import torch
        import json
        import os
        
        pc_str = ""
        if pc_size is not None:
            pc_str = f"_{pc_size}pc"

        file_dir = "data/local_data/celegans_processed"
        metadata_file = f"{file_dir}/celegans_metadata.csv"
        data_file = f"{file_dir}/celegans_proccessed{pc_str}.csv"
        
        metadata = pd.read_csv(metadata_file)
        data = pd.read_csv(data_file)
        X = data.iloc[:, 1:].values  # Remove first column (index)
        cell_types = metadata['cell_type'].values
        
        cell_types_str = [str(ct) if pd.notna(ct) else "NA" for ct in cell_types]
        unique_cell_types = sorted(list(set(cell_types_str)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_cell_types)}
        idx_to_label = {str(idx): label for idx, label in enumerate(unique_cell_types)}
        
        y = np.array([label_to_idx.get(str(ct) if pd.notna(ct) else "NA", -1) for ct in cell_types])
        valid_indices = y != -1
        X = X[valid_indices]
        y = y[valid_indices]
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
       
        
        if is_saving:
            label_mapping = {
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "num_classes": len(unique_cell_types),
                "unique_labels": unique_cell_types
            }

            # SAVE label mapping to JSON file!
            mapping_file = f"{file_dir}/label_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(label_mapping, f, indent=2)
            print(f"  Label mapping saved to: {mapping_file}")
        
        print(f"C. elegans Processed Dataset:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(unique_cell_types)}")
        print(f"  Class labels: {sorted(unique_cell_types)}")

        
        return X, y

    elif dataset_name == "scp2745":
        import numpy as np
        import torch
        X = np.load("data/local_data/SCP2745/X.npy")
        y = np.load("data/local_data/SCP2745/y.npy")

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        print(f"SCP2745 Dataset:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(torch.unique(y))}")
        print(f"  Class labels: {sorted(torch.unique(y).tolist())}")

        return X, y

    # make a noisy swiss roll embedded in 1000D
    elif dataset_name == "noisy_swiss_roll":
        import numpy as np
        n_dim = 1000
        n_data = 10000
        from sklearn.datasets import make_swiss_roll
        import torch
        
        X_low_dim, y = make_swiss_roll(n_samples=n_data, noise=0.5)

        # # plot the data in 3d
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(7, 5))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X_low_dim[:, 0], X_low_dim[:, 1], X_low_dim[:, 2], c=y, cmap='Spectral', s=5)
        # ax.set_title("Swiss Roll in 3D (latent space)")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.tight_layout()
        # plt.show()
        
        # Embed to 1000D: use random projection matrix
        np.random.seed(42)
        projection_matrix = np.random.randn(3, n_dim) / np.sqrt(3)
        X_high_dim = X_low_dim @ projection_matrix
        
        noise = np.random.randn(*X_high_dim.shape) * 0.1
        X_high_dim = X_high_dim + noise
        
        X = torch.tensor(X_high_dim, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        return X, y


    else:
        # ! read the data
        X, y = read_data(dataset_name, number=2000)
        print("!!LOADING DATA: X.shape, y.shape: ", dataset_name, X.shape, y.shape)
        sample_size = number
        X = X[:sample_size]
        y = y[:sample_size]

    import torch
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)


    return X, y



# testing with swiss role
if __name__ == "__main__":
    X, y = loading_toy_data("noisy_swiss_roll", number=10000)
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    print("np.unique(y): ", np.unique(y))
