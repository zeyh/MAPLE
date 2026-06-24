import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
from typing import Tuple


class MMCR_Loss(nn.Module):
    def __init__(self, lmbda: float, n_aug: int, distributed: bool = False):
        super(MMCR_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug
        self.distributed = distributed
        self.first_time = True
    
    # def _svdvals_mps_fix(self, tensor):
    #     #Move tensor to CPU for SVD if on MPS device, then move result back. For error that MAC M2 mps does not support SVD
    #     device = tensor.device
    #     dtype = tensor.dtype
    #     if device.type == "mps":
    #         result = torch.linalg.svdvals(tensor.cpu().float()).to(device=device, dtype=dtype)
    #     else:
    #         if dtype == torch.float16:
    #             result = torch.linalg.svdvals(tensor.float()).to(dtype=dtype)
    #         else:
    #             result = torch.linalg.svdvals(tensor)
    #     return result

    def _svdvals_mps_fix(self, tensor):
        original_device = tensor.device
        original_dtype = tensor.dtype
        
        # 2. NUCLEAR: FORCE CPU & FLOAT64
        #tensor_cpu = tensor.cpu().to(dtype=torch.float64)
        tensor_cpu = tensor.cpu().float()
        # COMPUTE GRAM MATRIX 
        # This fixes the "Size 128 vs 15" mismatch
        if tensor_cpu.dim() == 3:
            B, C, N = tensor_cpu.shape
            if C < N:
                # [B, C, N] * [B, N, C] -> [B, C, C]
                gram = torch.bmm(tensor_cpu, tensor_cpu.transpose(1, 2))
                size = C
            else:
                # [B, N, C] * [B, C, N] -> [B, N, N]
                gram = torch.bmm(tensor_cpu.transpose(1, 2), tensor_cpu)
                size = N
        else:
            # 2D Case 
            if tensor_cpu.shape[0] < tensor_cpu.shape[1]:
                gram = torch.mm(tensor_cpu, tensor_cpu.t())
                size = tensor_cpu.shape[0]
            else:
                gram = torch.mm(tensor_cpu.t(), tensor_cpu)
                size = tensor_cpu.shape[1]
        
        # ADD JITTER 
        epsilon = 1e-4
        eye = torch.eye(size, device=tensor_cpu.device, dtype=tensor_cpu.dtype)
        
        if gram.dim() == 3:
            eye = eye.unsqueeze(0).expand(gram.shape[0], -1, -1)
            
        gram = gram + epsilon * eye
        
        # EIGENVALUES -> SINGULAR VALUES
        eigenvalues = torch.linalg.eigvalsh(gram)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        S = torch.sqrt(eigenvalues)
        
        return S.to(device=original_device, dtype=original_dtype)



    def forward(self, z: Tensor, weights: Tensor = None) -> Tuple[Tensor, dict]:
        from config import CONFIG_MMCR

        z = F.normalize(z, dim=-1)
        z_local_ = einops.rearrange(z, "(B N) C -> B C N", N=self.n_aug)

        # gather across devices into list
        if self.distributed:
            z_list = [
                torch.zeros_like(z_local_)
                for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(z_list, z_local_, async_op=False)
            z_list[torch.distributed.get_rank()] = z_local_

            # append all
            z_local = torch.cat(z_list) 

        else:
            z_local = z_local_

        if weights is not None and CONFIG_MMCR["isSoft_MMCR_global"]: # ! attempts 1: pull the centroids towards the weighted mean of the neighbors
            ego_points = z_local[:, :, 0]  # [B, C, 1]
            neighbors = z_local[:, :, 1:]  # [B, C, 14]
            weights = weights.unsqueeze(1)  # [B, 1, 14]
            weighted_neighbors = torch.sum(neighbors * weights, dim=-1) / torch.sum(weights, dim=-1)  # [B, C]
            centroids = (ego_points + weighted_neighbors) / 2

        else:
            centroids = torch.mean(z_local, dim=-1)  # [B, C] # ! original

        if self.lmbda != 0.0:
            if weights is not None and CONFIG_MMCR["isSoft_MMCR_local"]:
                # weighted svd for each manifold but only for the neighbors
                ego_points = z_local[:, :, 0:1]  # [B, C, 1]
                neighbors = z_local[:, :, 1:]  # [B, C, 14]
                weights = weights.unsqueeze(1)
                weighted_neighbors = neighbors * weights  # [B, C, 14]
                weighted_z_local = torch.cat([ego_points, weighted_neighbors], dim=-1)  # [B, C, 15]
                local_nuc_vec = self._svdvals_mps_fix(weighted_z_local)
                local_nuc = local_nuc_vec.sum()
            else: 
                local_nuc_vec = self._svdvals_mps_fix(z_local)
                local_nuc = local_nuc_vec.sum() # ! original
        else:
            local_nuc = torch.tensor(0.0, device=z_local.device)
            local_nuc_vec = None


        global_nuc_vec = self._svdvals_mps_fix(centroids)
        global_nuc = global_nuc_vec.sum()

        batch_size = z_local.shape[0]

        #print(f"local_nuc: {local_nuc.item()}, global_nuc: {global_nuc.item()}")

        loss = self.lmbda * local_nuc / batch_size - global_nuc

        if self.lmbda != 0.0:
            loss_dict = {
                "loss": loss.item(),
                "local_nuc": local_nuc.item(),
                "global_nuc": global_nuc.item(),
                "local_nuc_vec": local_nuc_vec.cpu() if local_nuc_vec is not None else None,
                "global_nuc_vec": global_nuc_vec.cpu(),
            }
        else:
            loss_dict = {
                "loss": loss.item(),
                "global_nuc": global_nuc.item(),
                "global_nuc_vec": global_nuc_vec.cpu(),
            }
        
        # if CONFIG_MMCR.get("compute_pr_and_r", False):
        #     # z_local: [B, C, N] where B=batch, C=embedding_dim, N=neighbors+ego
        #     B, C, N = z_local.shape
        #     pr_per_point = torch.zeros(B, device=z_local.device)
        #     r_per_point = torch.zeros(B, device=z_local.device)
            
        #     # Center the data for each point: [B, C, N]
        #     z_local_mean = z_local.mean(dim=2, keepdim=True)  # [B, C, 1]
        #     z_local_centered = z_local - z_local_mean  # [B, C, N]
            
        #     # Compute covariance matrices for all points at once
        #     # For each batch: cov = (1/(N-1)) * z_centered @ z_centered^T
        #     if N > 1:
        #         cov_matrices = torch.bmm(z_local_centered, z_local_centered.transpose(1, 2)) / (N - 1)  # [B, C, C]
        #     else:
        #         cov_matrices = torch.zeros(B, C, C, device=z_local.device)
            
        #     # Compute eigenvalues for all covariance matrices
        #     eigenvals_all = torch.linalg.eigvalsh(cov_matrices)  # [B, C]
        #     # Only consider positive eigenvalues (numerical stability)
        #     eigenvals_all = torch.clamp(eigenvals_all, min=0.0)
            
        #     # Compute PR (D_M) = (sum λ)^2 / (sum λ^2) and R_M = sqrt(sum λ^2) for each point
        #     l1_norms = eigenvals_all.sum(dim=1)  # [B] - sum of eigenvalues
        #     l2_norms_sq = (eigenvals_all ** 2).sum(dim=1)  # [B] - sum of squared eigenvalues
            
        #     # PR = (l1_norm)^2 / l2_norm_sq
        #     pr_per_point = torch.where(
        #         l2_norms_sq > 1e-10,
        #         (l1_norms ** 2) / l2_norms_sq,
        #         torch.zeros_like(l1_norms)
        #     )  # [B]
            
        #     # R = sqrt(sum of squared eigenvalues) = sqrt(l2_norm_sq)
        #     r_per_point = torch.sqrt(l2_norms_sq)  # [B]
            
        #     loss_dict["pr_per_point"] = pr_per_point  # [B] - one PR value per point
        #     loss_dict["r_per_point"] = r_per_point  # [B] - one R value per point
        
        self.first_time = False

        return loss, loss_dict




# https://github.com/IgorSusmelj/barlowtwins/blob/main/loss.py
class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD

        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param

        loss = c_diff.sum()

        return loss


class RunningMean:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.value = None

    def update(self, x):
        x = x.detach().cpu().item()
        if self.value is None:
            self.value = x
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * x
        return self.value