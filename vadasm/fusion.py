"""
Subspace fusion and injection for V-ADASM
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SubspaceFuser:
    """Fuse vision subspaces into small models using TIES + DARE"""
    
    def __init__(self, fusion_beta=0.3, ties_drop_rate=0.3):
        self.fusion_beta = fusion_beta
        self.ties_drop_rate = ties_drop_rate
    
    def fuse_vision_deltas(self, small_layer, vision_deltas: torch.Tensor,
                          layer_idx: int) -> None:
        """
        Fuse vision deltas into small model layer using TIES + DARE
        
        Args:
            small_layer: Layer to modify
            vision_deltas: Vision deltas to fuse
            layer_idx: Layer index for selective fusion
        """
        logger.info(f"Fusing vision deltas into layer {layer_idx}...")
        
        # Identify target parameters to modify
        if hasattr(small_layer, 'mlp') and hasattr(small_layer.mlp, 'gate_proj'):
            target_param = small_layer.mlp.gate_proj.weight
        elif hasattr(small_layer, 'self_attn') and hasattr(small_layer.self_attn, 'q_proj'):
            target_param = small_layer.self_attn.q_proj.weight
        else:
            logger.warning(f"No suitable parameters found in layer {layer_idx}")
            return
        
        # Apply TIES + DARE fusion
        fused_delta = self._ties_dare_fusion(vision_deltas, target_param)
        
        # Inject fused deltas
        target_param.data.add_(fused_delta)
    
    def inject_vision_projector(self, model, projector_weights: torch.Tensor) -> None:
        """
        Inject vision projector for multimodal inference
        
        Args:
            model: Model to augment
            projector_weights: Trained/reduced projector weights
        """
        # Create projector layer
        projector = nn.Linear(
            projector_weights.shape[1], 
            projector_weights.shape[0]
        )
        projector.weight.data = projector_weights
        
        # Add as model attribute for inference
        model.vision_projector = projector
        
        # Update config
        if not hasattr(model, 'config'):
            model.config = type('Config', (), {})()
        
        model.config.has_vision = True
        model.config.vision_config = {
            "projector_dim": projector_weights.shape[1],
            "hidden_dim": projector_weights.shape[0]
        }
    
    def _ties_dare_fusion(self, vision_deltas: torch.Tensor, 
                         base_params: torch.Tensor) -> torch.Tensor:
        """
        Apply TIES (Trim, Elect Sign) + DARE (Drop And REscale) fusion
        
        Args:
            vision_deltas: Vision parameter deltas
            base_params: Original base model parameters
            
        Returns:
            Fused parameter delta
        """
        # Compute delta from vision subspace (simplified projection)
        beta = self.fusion_beta
        delta = beta * (vision_deltas - base_params)
        
        # TIES: Resolve sign conflicts
        base_sign = torch.sign(base_params)
        delta_sign = torch.sign(delta)
        
        # Conflict mask: where signs disagree
        conflict_mask = (base_sign != delta_sign) & (base_sign != 0)
        
        # Keep delta if it agrees or has larger magnitude
        abs_delta = torch.abs(delta)
        abs_base = torch.abs(base_params)
        
        keep_mask = ~conflict_mask | (abs_delta >= abs_base)
        ties_delta = torch.where(keep_mask, delta, torch.zeros_like(delta))
        
        # DARE: Drop small magnitudes and rescale survivors
        abs_ties = torch.abs(ties_delta).float()  # Convert to float for quantile
        threshold = torch.quantile(abs_ties, self.ties_drop_rate)
        dare_mask = torch.abs(ties_delta) > threshold
        
        # Rescale to preserve total magnitude
        kept_elements = dare_mask.sum()
        rescale_factor = ties_delta.numel() / kept_elements if kept_elements > 0 else 1.0
        
        final_delta = torch.where(dare_mask, ties_delta * rescale_factor, 
                                torch.zeros_like(ties_delta))
        
        return final_delta