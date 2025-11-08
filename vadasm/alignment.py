"""
Cross-modal alignment for V-ADASM
"""

import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CrossModalAligner:
    """Align text and vision representations across modalities"""
    
    def __init__(self, device="cuda"):
        self.device = device
    
    def align_neurons_by_similarity(self, small_acts: torch.Tensor, 
                                 large_acts: torch.Tensor,
                                 threshold: float = 0.8) -> torch.Tensor:
        """
        Align neurons using cosine similarity + Hungarian algorithm
        
        Args:
            small_acts: Activations from small model (batch x hidden)
            large_acts: Activations from large model (batch x hidden) 
            threshold: Similarity threshold for alignment
            
        Returns:
            Permutation tensor mapping large to small neurons
        """
        logger.info("Aligning neurons with Hungarian matching...")
        
        # Cosine similarity matrix
        cos_sim = cosine_similarity(
            small_acts.cpu().numpy(),
            large_acts.cpu().numpy()
        )
        
        # Hungarian algorithm (minimizes negative similarity)
        row_ind, col_ind = linear_sum_assignment(-cos_sim)
        
        # Create permutation tensor
        perm = torch.zeros(large_acts.shape[0], dtype=torch.long)
        perm[row_ind] = torch.tensor(col_ind, dtype=torch.long)
        
        return perm.to(self.device)
    
    def align_layer_permutations(self, small_model, large_model,
                               cross_acts: torch.Tensor,
                               alignment_ratio: float = 0.2) -> Dict[int, torch.Tensor]:
        """
        Compute permutations for each layer to align modalities
        
        Args:
            small_model: Small base model
            large_model: Large donor model  
            cross_acts: Cross-modal activation similarities
            alignment_ratio: Fraction of layers to align (from start)
            
        Returns:
            Dict of layer_idx -> permutation tensor
        """
        alignments = {}
        
        num_layers = len(large_model.layers)
        align_layers = int(num_layers * alignment_ratio)
        
        logger.info(f"Aligning first {align_layers}/{num_layers} layers...")
        
        for layer_idx in range(align_layers):
            small_layer = small_model.layers[layer_idx]
            large_layer = large_model.layers[layer_idx]
            
            # Get layer activations (simplified)
            if hasattr(small_layer, 'self_attn'):
                small_acts = self._get_attention_outputs(small_layer, cross_acts)
                large_acts = self._get_attention_outputs(large_layer, cross_acts)
            else:
                small_acts = self._get_mlp_outputs(small_layer, cross_acts)  
                large_acts = self._get_mlp_outputs(large_layer, cross_acts)
            
            # Compute permutation
            perm = self.align_neurons_by_similarity(small_acts, large_acts)
            alignments[layer_idx] = perm
        
        return alignments
    
    def pool_vision_patches(self, vision_features: torch.Tensor, 
                          target_dim: int) -> torch.Tensor:
        """
        Pool vision patch features to align with text dimensions
        
        Args:
            vision_features: Vision patch embeddings (batch x patches x dim)
            target_dim: Target text embedding dimension
            
        Returns:
            Pooled vision features
        """
        # Simple mean pooling + projection
        pooled = vision_features.mean(dim=1)  # Average over patches
        
        if pooled.shape[-1] != target_dim:
            # Linear projection to match dimensions
            projector = nn.Linear(pooled.shape[-1], target_dim).to(pooled.device)
            pooled = projector(pooled)
        
        return pooled
    
    def _get_attention_outputs(self, layer, inputs: torch.Tensor) -> torch.Tensor:
        """Extract attention layer outputs for alignment"""
        # Simplified - would use forward hooks
        return torch.randn(inputs.shape[0], layer.hidden_size).to(self.device)
    
    def _get_mlp_outputs(self, layer, inputs: torch.Tensor) -> torch.Tensor:
        """Extract MLP layer outputs for alignment"""  
        # Simplified
        return torch.randn(inputs.shape[0], layer.hidden_size).to(self.device)