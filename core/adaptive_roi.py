import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveROI(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        
    def forward(self, image):
        """Generate ROI based on attention"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0)
        
        # Extract patches
        patches = self.conv(image).flatten(2).permute(0,2,1)
        
        # Compute attention
        attn_weights, _ = self.attention(patches, patches, patches)
        
        # Get ROI from attention weights
        roi = self._weights_to_roi(attn_weights.mean(dim=1))
        return roi
    
    def _weights_to_roi(self, weights):
        """Convert attention weights to ROI coordinates"""
        # weights shape: [batch, num_patches]
        weights = weights.softmax(dim=-1)
        
        # patches are in 14x14 grid (for 224x224 input with 16x16 patches)
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, 14),
            torch.linspace(0, 1, 14)
        ), dim=-1).to(weights.device)
        
        # Compute weighted mean and std
        mean = (grid * weights.view(-1, 1, 1, 1)).sum(dim=(1,2))
        std = torch.sqrt(((grid - mean.view(-1, 1, 1, 2))**2 * weights.view(-1, 1, 1, 1)).sum(dim=(1,2)))
        
        # ROI is mean ± 2*std
        roi = torch.stack([
            torch.clamp(mean[:,0] - 2*std[:,0], 0),
            torch.clamp(mean[:,1] - 2*std[:,1], 0),
            torch.clamp(mean[:,0] + 2*std[:,0], 1),
            torch.clamp(mean[:,1] + 2*std[:,1], 1)
        ], dim=1)
        
        return roi.squeeze(0).cpu().numpy()