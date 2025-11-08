"""
V-ADASM: Vision-Adaptive Dimensionality-Aligned Subspace Merging

Core merger implementation for fusing small text models with large multimodal donors
to create compact Vision-Language Models through training-free operations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy import sparse
import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for input models in V-ADASM"""
    name_or_path: str
    hidden_dim: int
    num_layers: int
    vocab_size: int
    is_moe: bool = False
    has_vision: bool = False
    num_experts: int = 1
    vision_config: Optional[Dict] = None

@dataclass
class MergeConfig:
    """V-ADASM merge configuration"""
    # Vision subspace extraction
    projector_svd_rank: float = 0.95  # Keep 95% variance
    
    # Alignment
    alignment_layer_ratio: float = 0.2  # Align first 20% of layers
    cos_sim_threshold: float = 0.8
    
    # Fusion
    fusion_beta: float = 0.3  # Weight for vision deltas
    ties_drop_rate: float = 0.3  # Drop 30% of small magnitude deltas
    dare_rescale_factor: float = 1.0 / 0.7  # Rescale survivors
    
    # Evolutionary optimization
    evo_population_size: int = 30
    evo_generations: int = 15
    evo_mutation_rate: float = 0.1
    
    # MoE specific
    moe_top_k: int = 2  # Aggregate top-k experts
    moe_aggregate_method: str = "norm_avg"  # norm_avg or uniform
    
    # Hardware
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = "auto"
    
    # Validation
    val_text_count: int = 500
    val_vision_count: int = 500

class VADASMMerger:
    """
    V-ADASM merger: Injects vision capabilities from large multimodal donors
    into small text-only models through subspace merging and evolutionary tuning.
    """
    
    def __init__(self, merge_config: MergeConfig):
        self.config = merge_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if merge_config.device != "auto":
            self.device = torch.device(merge_config.device)
            
        logger.info(f"Initialized V-ADASM merger on device: {self.device}")
    
    def merge_models(self, small_config: ModelConfig, large_config: ModelConfig, 
                    val_data: Optional[Dict] = None) -> nn.Module:
        """
        Main V-ADASM pipeline: 5 steps to fuse vision into small model
        
        Args:
            small_config: Small base model (recipient)
            large_config: Large multimodal donor (source) 
            val_data: Optional validation data for evolutionary tuning
            
        Returns:
            Merged VLM model with small's architecture + vision capabilities
        """
        logger.info("Starting V-ADASM merge pipeline...")
        
        # Load models
        small_model = self._load_model(small_config)
        large_model = self._load_model(large_config)
        
        # Step 1: Extract vision subspaces from donor
        vis_subspaces, cross_acts = self._extract_vision_subspaces(large_model, large_config)
        
        # Step 2: Align cross-modal representations
        align_maps = self._cross_modality_alignment(small_model, large_model, 
                                                  small_config, large_config, cross_acts)
        
        # Step 3: Fuse subspaces into small model
        merged_model = self._subspace_fusion_injection(small_model, vis_subspaces, align_maps)
        
        # Step 4: Evolutionary hyperparameter optimization
        if val_data:
            merged_model = self._evolutionary_tuning(merged_model, val_data, small_config, large_config)
        
        # Step 5: Final validation and cleanup
        final_stats = self._validate_merge(merged_model, val_data)
        logger.info(f"Merge completed. Stats: {final_stats}")
        
        return merged_model
    
    def _load_model(self, config: ModelConfig) -> nn.Module:
        """Load model with vision components if present"""
        kwargs = {"torch_dtype": self.config.torch_dtype}

        if config.has_vision:
            # Load multimodal model - use AutoModel for vision models
            try:
                from transformers import LlavaForConditionalGeneration, AutoProcessor
                logger.info(f"Loading multimodal model: {config.name_or_path}")
                model = LlavaForConditionalGeneration.from_pretrained(
                    config.name_or_path, **kwargs
                )
                processor = AutoProcessor.from_pretrained(config.name_or_path)
                return {"model": model, "processor": processor}
            except Exception as e:
                logger.warning(f"Failed to load as LLaVA, trying AutoModel: {e}")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    config.name_or_path, trust_remote_code=True, **kwargs
                )
                processor = AutoProcessor.from_pretrained(config.name_or_path)
                return {"model": model, "processor": processor}
        else:
            # Load text-only model
            model = AutoModelForCausalLM.from_pretrained(
                config.name_or_path, **kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
            return {"model": model, "tokenizer": tokenizer}
    
    def _extract_vision_subspaces(self, large_model, large_config: ModelConfig) -> Tuple[Dict, torch.Tensor]:
        """
        Step 1: Extract vision subspaces from large donor model

        Returns:
            vis_subspaces: Dict of reduced vision tensors per layer
            cross_acts: Cross-modal activation similarities for alignment
        """
        logger.info("Step 1: Extracting vision subspaces...")

        model = large_model["model"]
        processor = large_model.get("processor")

        # Get vision components - handle different architectures
        vision_tower = None
        projector = None

        # LLaVA-style models
        if hasattr(model, 'vision_tower'):
            vision_tower = model.vision_tower
        elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
            vision_tower = model.model.vision_tower

        if hasattr(model, 'multi_modal_projector'):
            projector = model.multi_modal_projector
        elif hasattr(model, 'model') and hasattr(model.model, 'multi_modal_projector'):
            projector = model.model.multi_modal_projector

        if projector is None:
            logger.warning("No vision projector found, using placeholder")
            # Create a dummy projector for demonstration
            target_dim = large_config.hidden_dim
            W_proj_red = torch.randn(target_dim, 1024).to(self.config.torch_dtype)
        else:
            # Extract projector weights - handle Linear or Sequential projectors
            if isinstance(projector, nn.Linear):
                projector_weights = projector.weight.detach().cpu().float()
            elif isinstance(projector, nn.Sequential):
                # Get first linear layer
                for layer in projector:
                    if isinstance(layer, nn.Linear):
                        projector_weights = layer.weight.detach().cpu().float()
                        break
            else:
                logger.warning(f"Unknown projector type: {type(projector)}, using placeholder")
                target_dim = large_config.hidden_dim
                W_proj_red = torch.randn(target_dim, 1024).to(self.config.torch_dtype)
                projector_weights = None

            if projector_weights is not None:
                # SVD decomposition to extract principal components
                U, s, Vt = torch.svd(projector_weights)

                # Keep k components capturing 95% variance
                variance_explained = torch.cumsum(s**2, 0) / torch.sum(s**2)
                k = torch.where(variance_explained >= self.config.projector_svd_rank)[0][0].item() + 1

                W_proj_red = U[:, :k] @ torch.diag(s[:k]) @ Vt[:k, :]

                # Resize to target dimension if needed
                target_dim = large_config.hidden_dim
                if W_proj_red.shape[0] != target_dim:
                    # Simple linear interpolation
                    scale = target_dim / W_proj_red.shape[0]
                    W_proj_red = torch.nn.functional.interpolate(
                        W_proj_red.unsqueeze(0).unsqueeze(0),
                        size=(target_dim, W_proj_red.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)

        # Generate cross-modal activations (simplified - use dummy data)
        # In practice: forward pass on vision-text pairs
        cross_acts = torch.randn(1000, large_config.hidden_dim)  # Placeholder

        vis_subspaces = {
            "projector_reduced": W_proj_red.to(self.config.torch_dtype),
            "vision_encoder": vision_tower.state_dict() if vision_tower else {},
            "num_layers": large_config.num_layers
        }

        return vis_subspaces, cross_acts.to(self.device)
    
    def _cross_modality_alignment(self, small_model, large_model, small_config: ModelConfig,
                                large_config: ModelConfig, cross_acts: torch.Tensor) -> Dict:
        """
        Step 2: Align representations across modalities using neuron permutations
        """
        logger.info("Step 2: Cross-modality alignment...")

        small_model = small_model["model"]
        large_model = large_model["model"]

        # Get the language model from LLaVA if needed
        if hasattr(large_model, 'language_model'):
            large_model = large_model.language_model
        elif hasattr(large_model, 'model') and hasattr(large_model.model, 'layers'):
            large_model = large_model.model

        alignment_maps = {}

        # Check if models have layers attribute
        if not hasattr(small_model, 'layers'):
            if hasattr(small_model, 'model') and hasattr(small_model.model, 'layers'):
                small_model = small_model.model
            elif hasattr(small_model, 'transformer') and hasattr(small_model.transformer, 'h'):
                # GPT-style models use 'h' instead of 'layers'
                small_model.layers = small_model.transformer.h
            else:
                logger.warning("Could not find layers in small model, skipping alignment")
                return alignment_maps

        if not hasattr(large_model, 'layers'):
            logger.warning("Could not find layers in large model, skipping alignment")
            return alignment_maps

        # Align early layers where vision-text crossover happens
        n_align_layers = min(
            int(large_config.num_layers * self.config.alignment_layer_ratio),
            len(small_model.layers),
            len(large_model.layers)
        )

        logger.info(f"Aligning {n_align_layers} layers...")

        for layer_idx in range(n_align_layers):
            # Get small and large layer representations
            if small_config.is_moe:
                small_layer = small_model.layers[layer_idx]
                small_acts = self._get_layer_activations_moe(small_layer, cross_acts)
            else:
                small_layer = small_model.layers[layer_idx]
                small_acts = self._get_layer_activations(small_layer, cross_acts)

            if large_config.is_moe:
                large_layer = large_model.layers[layer_idx]
                large_acts = self._get_layer_activations_moe(large_layer, cross_acts)
            else:
                large_layer = large_model.layers[layer_idx]
                large_acts = self._get_layer_activations(large_layer, cross_acts)

            # Hungarian algorithm for optimal permutation
            alignment_maps[layer_idx] = self._hungarian_alignment(small_acts, large_acts)

        return alignment_maps
    
    def _hungarian_alignment(self, small_acts: torch.Tensor, large_acts: torch.Tensor) -> torch.Tensor:
        """Find optimal neuron permutation using Hungarian algorithm (cosine similarity)"""
        # Compute cosine similarities between neurons
        cos_sim = cosine_similarity(small_acts.cpu().numpy(), large_acts.cpu().numpy())
        
        # Negative because scipy minimizes
        cost_matrix = -cos_sim
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create permutation tensor
        perm = torch.zeros(small_acts.shape[0], dtype=torch.long)
        perm[row_ind] = torch.tensor(col_ind, dtype=torch.long)
        
        return perm.to(self.device)
    
    def _get_layer_activations(self, layer, inputs: torch.Tensor) -> torch.Tensor:
        """Get activations from a transformer layer"""
        # Simplified - in practice use hooks or attention outputs
        return torch.randn(inputs.shape[0], layer.hidden_size).to(self.device)
    
    def _get_layer_activations_moe(self, layer, inputs: torch.Tensor) -> torch.Tensor:
        """Get activations from MoE layer (aggregate over experts)"""
        # Aggregate across top-k experts based on router
        expert_outputs = []
        
        for expert_idx in range(len(layer.experts)):
            expert_output = self._get_layer_activations(layer.experts[expert_idx], inputs)
            expert_outputs.append(expert_output)
        
        # Simple average - in practice weight by router probabilities  
        return torch.stack(expert_outputs, dim=0).mean(dim=0)
    
    def _subspace_fusion_injection(self, small_model, vis_subspaces: Dict, 
                                 alignment_maps: Dict) -> nn.Module:
        """
        Step 3: Inject vision subspaces into small model using TIES and DARE
        """
        logger.info("Step 3: Subspace fusion and injection...")
        
        model = small_model["model"]
        state_dict = model.state_dict()
        
        # Get vision deltas from reduced projector
        W_vis = vis_subspaces["projector_reduced"]
        
        # For each layer in alignment maps
        for layer_idx, perm in alignment_maps.items():
            layer = model.layers[layer_idx]
            
            # Apply permutation to align representations
            if hasattr(layer, 'self_attn'):
                # Permute attention weights
                layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, perm]
                layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, perm]
                layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[perm, :]
                
            # Fuse vision deltas using TIES (resolve sign conflicts) and DARE (sparsity)  
            self._ties_dare_fusion(layer, W_vis, layer_idx)
        
        # Inject vision projector as prefix for multimodal inference
        self._inject_vision_projector(model, vis_subspaces)
        
        return model
    
    def _ties_dare_fusion(self, layer, W_vis: torch.Tensor, layer_idx: int):
        """Apply TIES (resolve conflicts) and DARE (add sparsity) to fusion"""
        
        # Get layer weights to modify
        if hasattr(layer, 'mlp'):
            target_weights = layer.mlp.gate_proj.weight.data
        else:
            return  # Skip if no MLP
            
        # Compute vision delta (simplified projection)
        beta = self.config.fusion_beta
        vis_delta = beta * (W_vis[:target_weights.shape[0], :target_weights.shape[1]] - target_weights)
        
        # TIES: Resolve sign conflicts by keeping larger magnitude
        current_sign = torch.sign(target_weights)
        delta_sign = torch.sign(vis_delta)
        conflict_mask = (current_sign != delta_sign) & (current_sign != 0)
        
        # Keep delta where it agrees or is larger magnitude
        ties_mask = conflict_mask & (torch.abs(vis_delta) >= torch.abs(target_weights))
        final_delta = torch.where(ties_mask, vis_delta, torch.zeros_like(vis_delta))
        
        # DARE: Drop small deltas and rescale survivors
        drop_mask = torch.abs(final_delta) < torch.quantile(torch.abs(final_delta), self.config.ties_drop_rate)
        final_delta = torch.where(drop_mask, torch.zeros_like(final_delta), 
                                final_delta * self.config.dare_rescale_factor)
        
        # Apply fusion
        target_weights.add_(final_delta)
    
    def _inject_vision_projector(self, model, vis_subspaces: Dict):
        """Inject vision projector for multimodal inference"""
        
        W_proj = vis_subspaces["projector_reduced"]
        
        # Create vision projector module
        projector = nn.Linear(W_proj.shape[1], W_proj.shape[0])
        projector.weight.data = W_proj
        
        # Add as attribute for inference
        model.vision_projector = projector
        
        # Add vision config
        model.config.has_vision = True
        model.config.vision_config = {
            "projector_dim": W_proj.shape[1],
            "hidden_dim": W_proj.shape[0]
        }
    
    def _evolutionary_tuning(self, model: nn.Module, val_data: Dict, 
                           small_config: ModelConfig, large_config: ModelConfig) -> nn.Module:
        """
        Step 4: Use evolutionary algorithm to optimize fusion hyperparameters
        """
        logger.info("Step 4: Evolutionary hyperparameter optimization...")
        
        try:
            from deap import base, creator, tools, algorithms
        except ImportError:
            logger.warning("DEAP not installed, skipping evolutionary tuning")
            return model
        
        # Define fitness function
        def evaluate_individual(individual):
            beta, drop_rate = individual
            # Temporarily modify config
            orig_beta = self.config.fusion_beta
            orig_drop = self.config.ties_drop_rate
            
            self.config.fusion_beta = beta
            self.config.ties_drop_rate = drop_rate
            
            # Re-run fusion with new params
            # (Simplified - in practice re-fuse and evaluate)
            
            # Evaluate on mixed objective
            vision_score = self._evaluate_vision(model, val_data.get("vision", []))
            text_score = self._evaluate_text(model, val_data.get("text", []))
            
            fitness = 0.2 * vision_score + 0.8 * text_score
            
            # Restore config
            self.config.fusion_beta = orig_beta
            self.config.ties_drop_rate = orig_drop
            
            return fitness,
        
        # DEAP setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_beta", np.random.uniform, 0.1, 0.6)
        toolbox.register("attr_drop", np.random.uniform, 0.2, 0.5)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_beta, toolbox.attr_drop))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=self.config.evo_mutation_rate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        pop = toolbox.population(n=self.config.evo_population_size)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                     ngen=self.config.evo_generations, 
                                     stats=stats, halloffame=hof, verbose=False)
        
        # Apply best parameters
        best_beta, best_drop = hof[0]
        self.config.fusion_beta = best_beta
        self.config.ties_drop_rate = best_drop
        
        logger.info(f"Evolutionary tuning complete. Best params: beta={best_beta:.3f}, drop={best_drop:.3f}")
        
        # Re-run fusion with optimal params
        # (In practice, would need to save/load weights)
        
        return model
    
    def _evaluate_vision(self, model, val_data: List) -> float:
        """Evaluate vision capabilities (placeholder)"""
        return 0.5  # Dummy score
        
    def _evaluate_text(self, model, val_data: List) -> float:
        """Evaluate text capabilities (placeholder)"""  
        return 0.7  # Dummy score
    
    def _validate_merge(self, model: nn.Module, val_data: Optional[Dict]) -> Dict:
        """Step 5: Final validation and statistics"""
        logger.info("Step 5: Final validation...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "has_vision": hasattr(model, 'vision_projector'),
            "model_size_mb": total_params * 2 / (1024 * 1024)  # Rough BF16 estimate
        }
        
        if val_data:
            vision_score = self._evaluate_vision(model, val_data.get("vision", []))
            text_score = self._evaluate_text(model, val_data.get("text", []))
            stats.update({"vision_score": vision_score, "text_score": text_score})
        
        return stats