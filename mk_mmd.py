"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Sequence
import torch
import torch.nn as nn
import numpy


__all__ = ['MultipleKernelMaximumMeanDiscrepancy']

class GaussianKernel(nn.Module):
    """Gaussian Kernel for MK-MMD computation"""
    def __init__(self, alpha: float = 1.0):
        super(GaussianKernel, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        # Compute pairwise squared distances
        x_norm = (x ** 2).sum(dim=1).view(n, 1) # chuẩn 2
        dist = x_norm + x_norm.view(1, n) - 2.0 * torch.mm(x, x.transpose(0, 1))
        # Apply Gaussian kernel
        return torch.exp(-self.alpha * dist)


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """Fixed MK-MMD implementation that handles different sample sizes"""
    
    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        self.cached_n_s = None
        self.cached_n_t = None

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # Flatten the inputs if they are multi-dimensional
        if z_s.dim() > 2:
            z_s = z_s.view(z_s.size(0), -1)
        if z_t.dim() > 2:
            z_t = z_t.view(z_t.size(0), -1)

        features = torch.cat([z_s, z_t], dim=0)  # features.shape = (n_s + n_t, d)
        n_s = int(z_s.size(0))
        n_t = int(z_t.size(0))
        
        # Update index matrix only if sizes changed
        if (self.index_matrix is None or 
            self.cached_n_s != n_s or 
            self.cached_n_t != n_t or
            self.index_matrix.device != z_s.device):
            
            self.index_matrix = self._update_index_matrix(n_s, n_t, self.linear).to(z_s.device)
            self.cached_n_s = n_s
            self.cached_n_t = n_t

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        loss = (kernel_matrix * self.index_matrix).sum()
        
        # Add regularization term for non-linear version
        # if not self.linear:
        #     loss = loss + 2. / float(min(n_s, n_t) - 1) if min(n_s, n_t) > 1 else loss

        return loss

    def _update_index_matrix(self, n_s: int, n_t: int, linear: Optional[bool] = True) -> torch.Tensor:
        """Update the index matrix for MK-MMD computation"""
        total_size = n_s + n_t
        index_matrix = torch.zeros(total_size, total_size)
        
        if linear:
            # Linear version: use circular pairing strategy
            for i in range(n_s):
                j = (i + 1) % n_s
                index_matrix[i, j] = 1. / float(n_s)
                
            for i in range(n_t):
                j = (i + 1) % n_t
                index_matrix[n_s + i, n_s + j] = 1. / float(n_t)
                
            # Source-target cross pairs (negative)
            for i in range(n_s):
                for j in range(n_t):
                    index_matrix[i, n_s + j] = -1. / float(n_s)
                    index_matrix[n_s + j, i] = -1. / float(n_t)
        else:
            # Non-linear version: all pairs
            for i in range(n_s):
                for j in range(n_s):
                    if i != j:
                        index_matrix[i, j] = 1. / float(n_s * (n_s - 1))
                        
            for i in range(n_t):
                for j in range(n_t):
                    if i != j:
                        index_matrix[n_s + i, n_s + j] = 1. / float(n_t * (n_t - 1))
                        
            # Source-target cross pairs (negative)
            for i in range(n_s):
                for j in range(n_t):
                    index_matrix[i, n_s + j] = -2. / float(n_s * n_t)
                    index_matrix[n_s + j, i] = -2. / float(n_s * n_t)
        
        return index_matrix
    