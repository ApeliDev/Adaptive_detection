import torch
import torch.nn as nn
import torch.nn.functional as F

class NASController(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        
        # Architecture search space
        self.search_space = {
            'backbone': ['yolo', 'detr', 'ensemble'],
            'resolution': [0.5, 0.75, 1.0],
            'roi_strategy': ['attention', 'saliency', 'fixed']
        }
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Policy heads
        self.backbone_head = nn.Linear(hidden_dim, 3)  # 3 backbone choices
        self.resolution_head = nn.Linear(hidden_dim, 3)  # 3 resolution choices
        self.roi_head = nn.Linear(hidden_dim, 3)  # 3 ROI strategies
        
        # Temperature for Gumbel-Softmax
        self.temperature = 1.0
    
    def forward(self, x):
        # Encode features
        features = self.encoder(x)
        
        # Get architecture logits
        backbone_logits = self.backbone_head(features)
        resolution_logits = self.resolution_head(features)
        roi_logits = self.roi_head(features)
        
        # Sample architectures using Gumbel-Softmax
        backbone = F.gumbel_softmax(backbone_logits, tau=self.temperature, hard=True)
        resolution = F.gumbel_softmax(resolution_logits, tau=self.temperature, hard=True)
        roi = F.gumbel_softmax(roi_logits, tau=self.temperature, hard=True)
        
        return {
            'backbone': backbone,
            'resolution': resolution,
            'roi_strategy': roi
        }
    
    def get_architecture(self, x):
        """Get human-readable architecture"""
        arch = self.forward(x)
        return {
            'backbone': self.search_space['backbone'][torch.argmax(arch['backbone'])],
            'resolution': self.search_space['resolution'][torch.argmax(arch['resolution'])],
            'roi_strategy': self.search_space['roi_strategy'][torch.argmax(arch['roi_strategy'])]
        }


        