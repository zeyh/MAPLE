# config/default_config.py
import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_CONFIG = {
    "name": "fashion_mnist_full", 
    "size": 40000,     
    "supported_datasets": [
        "s2",
        "s3",
        "s4",
        "s5",
        "s_curve",
        "cifar10",
        "fashion_mnist",
        "mnist_test",
        "digits",
        "toy",
        "wine",
        "high_dim",
        "low_dim_stacked",
        "checkerboard",
        "roll",
        "breast_cancer",
        "bank",
        "hatespeech",
        "cnae9",
        "coil20",
        "epileptic",
        "fmd",
        "har",
        "hiva", 
        "imdb",
        "orl",
    ]
}

# * =====================================================
_DEFAULT_CONFIG_SET0 = {
    "Projection_dim": 128, # ? test 128,256,512,1024
    "batch_size": 1024, # ! unchanged # test 512,1024,2048,4096
    "learning_rate": 0.001, # ! unchanged
    "weight_decay": 1e-6, # ! unchanged
    "KNN_Embedding_dim": 512, # ! NOT USED!
}

_DEFAULT_CONFIG_SET1 = {
    "n_neighbors": 15, # ? test 5, 10, 15, 30, 60, 100
    "epochs": 20, # ? test 10,20,30,40,50,60
}

_DEFAULT_CONFIG_SET2 = {
    "use_bt_loss": "False", # "Both", "True", "False" # ? test "True","False"
    "update_knn_every": 20, # ? test depend of epochs 10,20,30,40,50,60 
    "mu_temperature": 0.1,  # ? test 0.1,0.3,0.5,0.7,0.9
    
    "use_gpu_knn": True,  # ! unchanged # Use GPU-accelerated KNN computation (faster for large datasets)
    "use_amp": True,  # ! unchanged # Use Automatic Mixed Precision (AMP) for faster training
    "bt_loss_weight": 1, # ! NOT USED!
    "mmcr_loss_weight": 1, # ! NOT USED!
    "manual_recompute_knn_on_Z": False, # ! NOT USED!
    "adaptive_mu_temperature": False, # ! NOT USED!
    "use_dimension_shift": False,  # ! NOT USED! # If True, performs one-time shift from raw_X to z_all after first KNN rebuild
    "shift_warmup_epochs": 5,  # ! NOT USED! # Number of epochs with reduced LR after dimension shift
}

_DEFAULT_CONFIG_BT = {
    "use_learned_mu_for_view2": False, # ! unchanged 
    "view2_mode": "cosine", # ! unchanged # if use_learned_mu_for_view2 = False "mean", "pca", else only with cosine sim 
}

_DEFAULT_CONFIG_MMCR = {
    "lmbda": 0.1, # ? test 0.1,0.3,0.5,0.7,0.9
    "isSoft_MMCR_global": False, # ! NOT USED!
    "isSoft_MMCR_local": False, # ! NOT USED!
    "compute_pr_and_r": False,   # ! NOT USED!
}


# Global config variables 
CONFIG_SET0 = _DEFAULT_CONFIG_SET0.copy()
CONFIG_SET1 = _DEFAULT_CONFIG_SET1.copy()
CONFIG_SET2 = _DEFAULT_CONFIG_SET2.copy()
CONFIG_BT = _DEFAULT_CONFIG_BT.copy()
CONFIG_MMCR = _DEFAULT_CONFIG_MMCR.copy()


def set_test_config(test_params=None):
    global CONFIG_SET0, CONFIG_SET1, CONFIG_SET2, CONFIG_BT, CONFIG_MMCR
    
    if test_params is None:
        # Reset to defaults
        CONFIG_SET0 = _DEFAULT_CONFIG_SET0.copy()
        CONFIG_SET1 = _DEFAULT_CONFIG_SET1.copy()
        CONFIG_SET2 = _DEFAULT_CONFIG_SET2.copy()
        CONFIG_BT = _DEFAULT_CONFIG_BT.copy()
        CONFIG_MMCR = _DEFAULT_CONFIG_MMCR.copy()
        return
    
    # Override with test parameters
    for key, value in test_params.items():
        if key in CONFIG_SET0:
            CONFIG_SET0[key] = value
        elif key in CONFIG_SET1:
            CONFIG_SET1[key] = value
        elif key in CONFIG_SET2:
            CONFIG_SET2[key] = value
        elif key in CONFIG_BT:
            CONFIG_BT[key] = value
        elif key in CONFIG_MMCR:
            CONFIG_MMCR[key] = value
        else:
            print(f"Warning: Parameter '{key}' not found in any config set")


def reset_config():
    set_test_config(None)



