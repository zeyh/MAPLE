import matplotlib.pyplot as plt
import numpy as np
np.infty = np.inf 
import umap
import torch
from config.default_config import CONFIG_SET1


def scatter_plot(embeddings_2d, labels=None, title="umap", filename="umap_visualization", is_saving = True):
    is_3d = embeddings_2d.shape[1] == 3
    
    if is_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                embeddings_2d[:, 2],
                c=labels,
                cmap='tab10',
                alpha=0.6
            )
            plt.colorbar(scatter)
        else:
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                embeddings_2d[:, 2],
                alpha=0.6
            )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6
            )
            plt.colorbar(scatter)
        else:
            plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                alpha=0.6
            )
        
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.title(title)
    
    if is_saving:
        plt.savefig(f'knn_models/{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_embeddings(embeddings, labels=None, title="umap", filename="umap_visualization", is_saving = True):
    embeddings_np = embeddings.numpy()
    
    reducer = umap.UMAP(
        n_neighbors=CONFIG_SET1["n_neighbors"],
        min_dist=0.1,
        n_components=3,
        random_state=42,
    )
    
    embedding_2d = reducer.fit_transform(embeddings_np)
    scatter_plot(embedding_2d, labels, title, filename, False)
    
    return embedding_2d


def vis_2_embedings(embeddings1, embeddings2, labels=None, title1="plot1", title2="plot2", filename="vis", need_dr=False, is_saving=True, n_neighbors=15):
    # Convert tensors to numpy arrays if needed
    embeddings1_np = embeddings1.numpy() if isinstance(embeddings1, torch.Tensor) else embeddings1
    embeddings2_np = embeddings2.numpy() if isinstance(embeddings2, torch.Tensor) else embeddings2
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    if need_dr:
        embedding1_2d = reducer.fit_transform(embeddings1_np)
    else:
        embedding1_2d = embeddings1_np
    
    if need_dr:
        embedding2_2d = reducer.fit_transform(embeddings2_np)
    else:
        embedding2_2d = embeddings2_np
    
    if labels is not None:
        # Plot with labels
        point_size = 10
        if len(labels) >= 10000:
            point_size = 2
        if len(labels.shape) > 1:
            if labels.shape[1] > 1:
                single_labels = torch.zeros(labels.shape[0], dtype=torch.long)
                for i in range(labels.shape[0]):
                    non_zero_indices = torch.nonzero(labels[i]).flatten()
                    if len(non_zero_indices) > 0:
                        single_labels[i] = non_zero_indices[0]
                    else:
                        single_labels[i] = 0
                labels = single_labels

        num_unique_labels = len(np.unique(labels))
        if num_unique_labels <= 10:
            cmap = 'tab10'
        elif num_unique_labels <= 20:
            cmap = 'tab20'
        elif num_unique_labels <= 40:
            cmap = 'tab20b'
        elif num_unique_labels <= 64:
            cmap = 'tab20c'
        else:
            cmap = 'viridis'

        scatter1 = ax1.scatter(embedding1_2d[:, 0], embedding1_2d[:, 1], c=labels, alpha=0.6, s=point_size, cmap=cmap)
        scatter2 = ax2.scatter(embedding2_2d[:, 0], embedding2_2d[:, 1], c=labels, alpha=0.6, s=point_size, cmap=cmap)

        plt.colorbar(scatter1, ax=ax1)
        plt.colorbar(scatter2, ax=ax2)
    else:
        ax1.scatter(embedding1_2d[:, 0], embedding1_2d[:, 1], alpha=0.6, s=10)
        ax2.scatter(embedding2_2d[:, 0], embedding2_2d[:, 1], alpha=0.6, s=10)
    
    ax1.set_title(title1)
    ax2.set_title(title2)
    plt.suptitle("Embedding Comparison", fontsize=16)
    plt.tight_layout()
    
    if is_saving:
        import os
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return embedding1_2d, embedding2_2d
