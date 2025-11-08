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
    fusion_beta: float = 0.3  # Weight for vision deltas AND language knowledge transfer
    ties_drop_rate: float = 0.3  # Drop 30% of small magnitude deltas
    dare_rescale_factor: float = 1.0 / 0.7  # Rescale survivors

    # Language knowledge transfer
    language_transfer_enabled: bool = True  # Enable transferring language knowledge from large model
    language_transfer_beta: float = 0.3  # Weight for language knowledge (same as fusion_beta by default)

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
        Main V-ADASM pipeline: Enhanced 6-step merge for vision + language knowledge transfer

        Args:
            small_config: Small base model (recipient)
            large_config: Large multimodal donor (source)
            val_data: Optional validation data for evolutionary tuning

        Returns:
            Merged VLM model with small's architecture + vision capabilities + large model knowledge
        """
        logger.info("Starting Enhanced V-ADASM merge pipeline (vision + language knowledge)...")

        # Load models
        small_model = self._load_model(small_config)
        large_model = self._load_model(large_config)

        # Step 1: Extract vision subspaces from donor
        vis_subspaces, cross_acts = self._extract_vision_subspaces(large_model, large_config)

        # Step 2: Extract language knowledge deltas from large model
        lang_deltas = self._extract_language_knowledge(small_model, large_model, small_config, large_config)

        # Step 3: Align cross-modal representations
        align_maps = self._cross_modality_alignment(small_model, large_model,
                                                  small_config, large_config, cross_acts)

        # Step 4: Fuse both vision subspaces AND language knowledge into small model
        merged_model = self._subspace_fusion_injection(small_model, vis_subspaces, align_maps, lang_deltas)

        # Step 5: Evolutionary hyperparameter optimization
        if val_data:
            merged_model = self._evolutionary_tuning(merged_model, val_data, small_config, large_config)

        # Step 6: Final validation and cleanup
        final_stats = self._validate_merge(merged_model, val_data)
        logger.info(f"Merge completed. Stats: {final_stats}")

        return merged_model
    
    def _load_model(self, config: ModelConfig) -> nn.Module:
        """Load model with vision components if present - supports any model architecture"""
        kwargs = {"dtype": self.config.torch_dtype, "trust_remote_code": True}

        logger.info(f"Loading model: {config.name_or_path}")

        if config.has_vision:
            # Load multimodal model - try multiple approaches for maximum compatibility
            model = None
            processor = None

            try:
                from transformers import AutoProcessor

                # Import all known VLM classes (gracefully handle missing ones)
                vlm_classes = []

                # LLaVA family
                try:
                    from transformers import LlavaForConditionalGeneration
                    vlm_classes.append(("LlavaForConditionalGeneration", LlavaForConditionalGeneration))
                except ImportError:
                    pass

                try:
                    from transformers import LlavaNextForConditionalGeneration
                    vlm_classes.append(("LlavaNextForConditionalGeneration", LlavaNextForConditionalGeneration))
                except ImportError:
                    pass

                # Qwen-VL family (Qwen2VL, Qwen3VL)
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    vlm_classes.append(("Qwen2VLForConditionalGeneration", Qwen2VLForConditionalGeneration))
                except ImportError:
                    pass

                # Note: Qwen3VL might use same class as Qwen2VL or AutoModel
                # MiniCPM-V family
                try:
                    from transformers import MiniCPMV
                    vlm_classes.append(("MiniCPMV", MiniCPMV))
                except ImportError:
                    pass

                # CogVLM family
                try:
                    from transformers import CogVLMForCausalLM
                    vlm_classes.append(("CogVLMForCausalLM", CogVLMForCausalLM))
                except ImportError:
                    pass

                # InternVL family
                try:
                    from transformers import InternVLChatModel
                    vlm_classes.append(("InternVLChatModel", InternVLChatModel))
                except ImportError:
                    pass

                # Generic VLM classes
                try:
                    from transformers import AutoModelForVision2Seq
                    vlm_classes.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
                except ImportError:
                    pass

                # Paligemma, Idefics, etc.
                try:
                    from transformers import PaliGemmaForConditionalGeneration
                    vlm_classes.append(("PaliGemmaForConditionalGeneration", PaliGemmaForConditionalGeneration))
                except ImportError:
                    pass

                try:
                    from transformers import Idefics2ForConditionalGeneration
                    vlm_classes.append(("Idefics2ForConditionalGeneration", Idefics2ForConditionalGeneration))
                except ImportError:
                    pass

                # Try each model class
                last_error = None
                for class_name, model_class in vlm_classes:
                    try:
                        logger.info(f"Trying to load as {class_name}...")
                        model = model_class.from_pretrained(config.name_or_path, **kwargs)
                        logger.info(f"✓ Successfully loaded as {class_name}")
                        break
                    except Exception as e:
                        last_error = e
                        logger.debug(f"Failed to load as {class_name}: {str(e)[:100]}")
                        continue

                if model is None:
                    # Final fallback: AutoModel with trust_remote_code
                    logger.info("Trying AutoModel as final fallback...")
                    try:
                        model = AutoModel.from_pretrained(config.name_or_path, **kwargs)
                        logger.info("✓ Successfully loaded with AutoModel")
                    except Exception as e:
                        logger.error(f"All loading attempts failed. Last error: {e}")
                        raise RuntimeError(
                            f"Could not load multimodal model {config.name_or_path}. "
                            f"Tried {len(vlm_classes)} model classes and AutoModel. "
                            f"Last error: {str(last_error)[:200]}"
                        )

                # Load processor
                processor = AutoProcessor.from_pretrained(config.name_or_path, trust_remote_code=True)
                return {"model": model, "processor": processor}

            except Exception as e:
                logger.error(f"Failed to load multimodal model: {e}")
                raise

        else:
            # Load text-only model - supports all causal LM architectures
            try:
                model = AutoModelForCausalLM.from_pretrained(config.name_or_path, **kwargs)
                tokenizer = AutoTokenizer.from_pretrained(config.name_or_path, trust_remote_code=True)
                logger.info(f"Loaded text-only model: {type(model).__name__}")
                return {"model": model, "tokenizer": tokenizer}
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                raise
    
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
            # Extract projector weights - handle different projector types
            projector_weights = None

            if isinstance(projector, nn.Linear):
                projector_weights = projector.weight.detach().cpu().float()
            elif isinstance(projector, nn.Sequential):
                # Get first linear layer from Sequential
                for layer in projector:
                    if isinstance(layer, nn.Linear):
                        projector_weights = layer.weight.detach().cpu().float()
                        break
            elif hasattr(projector, 'linear_1'):
                # LlavaMultiModalProjector has linear_1, linear_2
                projector_weights = projector.linear_1.weight.detach().cpu().float()
                logger.info("Using LlavaMultiModalProjector.linear_1")
            elif hasattr(projector, 'layers') and len(projector.layers) > 0:
                # Some projectors have a 'layers' attribute
                first_layer = projector.layers[0]
                if isinstance(first_layer, nn.Linear):
                    projector_weights = first_layer.weight.detach().cpu().float()
            else:
                # Try to iterate through children
                for child in projector.children():
                    if isinstance(child, nn.Linear):
                        projector_weights = child.weight.detach().cpu().float()
                        break

            if projector_weights is None:
                logger.warning(f"Could not extract weights from projector type: {type(projector)}, using placeholder")
                target_dim = large_config.hidden_dim
                W_proj_red = torch.randn(target_dim, 1024).to(self.config.torch_dtype)

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

    def _extract_language_knowledge(self, small_model, large_model,
                                   small_config: ModelConfig, large_config: ModelConfig) -> Dict:
        """
        Step 2: Extract language knowledge deltas from large model's text capabilities

        This transfers the superior language understanding from the large model to the small model
        by computing parameter deltas for each layer.

        Returns:
            lang_deltas: Dict mapping layer_idx -> {param_name: delta_tensor}
        """
        if not self.config.language_transfer_enabled:
            logger.info("Step 2: Language knowledge transfer disabled, skipping...")
            return {}

        logger.info("Step 2: Extracting language knowledge from large model...")

        small_m = small_model["model"]
        large_m = large_model["model"]

        # Extract the language model from multimodal wrapper if needed
        if hasattr(large_m, 'language_model'):
            large_m = large_m.language_model
        elif hasattr(large_m, 'model') and hasattr(large_m.model, 'layers'):
            large_m = large_m.model

        # Get small model layers
        if hasattr(small_m, 'model') and hasattr(small_m.model, 'layers'):
            small_m = small_m.model
        elif hasattr(small_m, 'transformer') and hasattr(small_m.transformer, 'h'):
            small_m.layers = small_m.transformer.h

        if not hasattr(small_m, 'layers') or not hasattr(large_m, 'layers'):
            logger.warning("Could not find layers in models, skipping language knowledge extraction")
            return {}

        lang_deltas = {}

        # Extract deltas for each layer (align by layer index)
        num_layers = min(len(small_m.layers), len(large_m.layers))
        logger.info(f"Extracting language deltas for {num_layers} layers...")

        for layer_idx in range(num_layers):
            small_layer = small_m.layers[layer_idx]
            large_layer = large_m.layers[layer_idx]

            layer_deltas = {}

            # Extract deltas from attention weights
            if hasattr(small_layer, 'self_attn') and hasattr(large_layer, 'self_attn'):
                small_attn = small_layer.self_attn
                large_attn = large_layer.self_attn

                # Q, K, V projection deltas
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(small_attn, proj_name) and hasattr(large_attn, proj_name):
                        small_weight = getattr(small_attn, proj_name).weight.data
                        large_weight = getattr(large_attn, proj_name).weight.data

                        # Compute delta (with dimension handling)
                        delta = self._compute_weight_delta(small_weight, large_weight)
                        if delta is not None:
                            layer_deltas[f'self_attn.{proj_name}'] = delta

            # Extract deltas from MLP/FFN weights
            if hasattr(small_layer, 'mlp') and hasattr(large_layer, 'mlp'):
                small_mlp = small_layer.mlp
                large_mlp = large_layer.mlp

                # Gate, up, down projections (LLaMA-style)
                for proj_name in ['gate_proj', 'up_proj', 'down_proj', 'fc_in', 'fc_out']:
                    if hasattr(small_mlp, proj_name) and hasattr(large_mlp, proj_name):
                        small_weight = getattr(small_mlp, proj_name).weight.data
                        large_weight = getattr(large_mlp, proj_name).weight.data

                        delta = self._compute_weight_delta(small_weight, large_weight)
                        if delta is not None:
                            layer_deltas[f'mlp.{proj_name}'] = delta

            if layer_deltas:
                lang_deltas[layer_idx] = layer_deltas
                logger.debug(f"Layer {layer_idx}: extracted {len(layer_deltas)} parameter deltas")

        logger.info(f"✓ Extracted language knowledge from {len(lang_deltas)} layers")
        return lang_deltas

    def _compute_weight_delta(self, small_weight: torch.Tensor, large_weight: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute parameter delta between large and small model weights
        Handles dimension mismatches via projection/slicing
        """
        # Handle dimension mismatch
        if small_weight.shape != large_weight.shape:
            # If shapes completely incompatible, skip
            if small_weight.dim() != large_weight.dim():
                return None

            # Slice or pad to match small model dimensions
            delta = torch.zeros_like(small_weight)

            # Determine overlap region
            min_dim0 = min(small_weight.shape[0], large_weight.shape[0])
            min_dim1 = min(small_weight.shape[1], large_weight.shape[1])

            # Compute delta for overlapping region
            small_slice = small_weight[:min_dim0, :min_dim1]
            large_slice = large_weight[:min_dim0, :min_dim1]

            delta[:min_dim0, :min_dim1] = large_slice - small_slice
            return delta.detach().cpu()

        # Same shape - direct delta
        delta = large_weight - small_weight
        return delta.detach().cpu()

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

        # Handle dimension mismatch - project to common dimension
        if small_acts.shape[1] != large_acts.shape[1]:
            logger.info(f"Aligning dimensions: small={small_acts.shape[1]}, large={large_acts.shape[1]}")

            # Project to smaller dimension to preserve information
            target_dim = min(small_acts.shape[1], large_acts.shape[1])

            if small_acts.shape[1] > target_dim:
                # Project small acts down
                projection = torch.randn(small_acts.shape[1], target_dim).to(small_acts.device)
                small_acts = small_acts @ projection

            if large_acts.shape[1] > target_dim:
                # Project large acts down
                projection = torch.randn(large_acts.shape[1], target_dim).to(large_acts.device)
                large_acts = large_acts @ projection

        # Ensure we have enough samples for alignment
        n_samples = min(small_acts.shape[0], large_acts.shape[0])
        if small_acts.shape[0] != large_acts.shape[0]:
            small_acts = small_acts[:n_samples]
            large_acts = large_acts[:n_samples]

        # Compute cosine similarities between neurons
        cos_sim = cosine_similarity(small_acts.cpu().numpy(), large_acts.cpu().numpy())

        # Negative because scipy minimizes
        cost_matrix = -cos_sim

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create permutation tensor
        perm = torch.zeros(n_samples, dtype=torch.long)
        perm[row_ind] = torch.tensor(col_ind, dtype=torch.long)

        return perm.to(self.device)
    
    def _get_layer_activations(self, layer, inputs: torch.Tensor) -> torch.Tensor:
        """Get activations from a transformer layer"""
        # Get hidden size from layer config or attributes
        if hasattr(layer, 'hidden_size'):
            hidden_size = layer.hidden_size
        elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'hidden_size'):
            hidden_size = layer.self_attn.hidden_size
        elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'embed_dim'):
            hidden_size = layer.self_attn.embed_dim
        elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn.q_proj, 'out_features'):
            hidden_size = layer.self_attn.q_proj.out_features
        else:
            # Fallback to input dimension
            hidden_size = inputs.shape[-1]

        # Simplified - in practice use hooks or attention outputs
        return torch.randn(inputs.shape[0], hidden_size).to(self.device)
    
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
                                 alignment_maps: Dict, lang_deltas: Dict) -> nn.Module:
        """
        Step 4: Inject both vision subspaces AND language knowledge into small model using TIES and DARE

        This method now does:
        1. Applies permutations from alignment
        2. Injects vision capabilities via TIES+DARE fusion
        3. Transfers language knowledge from large model via TIES+DARE on deltas
        """
        logger.info("Step 4: Subspace fusion and injection (vision + language)...")

        model = small_model["model"]
        state_dict = model.state_dict()

        # Get the actual model layers
        if hasattr(model, 'layers'):
            layers = model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            logger.warning("Could not find model layers, skipping fusion")
            # Still inject vision projector
            self._inject_vision_projector(model, vis_subspaces)
            return model

        # Get vision deltas from reduced projector
        W_vis = vis_subspaces["projector_reduced"]

        # For each layer in alignment maps
        for layer_idx, perm in alignment_maps.items():
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range, skipping")
                continue

            layer = layers[layer_idx]

            # Apply permutation to align representations (if perm is valid)
            if hasattr(layer, 'self_attn') and len(perm) > 0:
                try:
                    # Ensure perm is on the same device as the weights
                    device = layer.self_attn.q_proj.weight.device
                    perm = perm.to(device)

                    # Permute attention weights
                    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, perm]
                    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, perm]
                    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[perm, :]
                except Exception as e:
                    logger.warning(f"Could not permute layer {layer_idx}: {e}")

            # Fuse vision deltas using TIES (resolve sign conflicts) and DARE (sparsity)
            self._ties_dare_fusion(layer, W_vis, layer_idx)

        # Inject language knowledge from large model
        logger.info("Injecting language knowledge deltas...")
        num_lang_injections = 0
        for layer_idx in range(len(layers)):
            if layer_idx in lang_deltas:
                layer = layers[layer_idx]
                layer_deltas = lang_deltas[layer_idx]

                # Apply each parameter delta using TIES+DARE
                for param_path, delta in layer_deltas.items():
                    self._apply_language_delta(layer, param_path, delta)
                    num_lang_injections += 1

        logger.info(f"✓ Applied {num_lang_injections} language knowledge deltas across {len(lang_deltas)} layers")

        # Inject vision projector as prefix for multimodal inference
        self._inject_vision_projector(model, vis_subspaces)

        return model
    
    def _compute_quantile_safe(self, tensor: torch.Tensor, q: float, max_elements: int = 10_000_000):
        """
        Safely compute quantile, sampling if tensor is too large
        
        Args:
            tensor: Input tensor (should be 1D or will be flattened)
            q: Quantile to compute (0.0 to 1.0)
            max_elements: Maximum elements before sampling
            
        Returns:
            Quantile value
        """
        flat_tensor = tensor.flatten()
        n_elements = flat_tensor.numel()
        
        if n_elements == 0:
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        
        if n_elements <= max_elements:
            # Small enough, compute directly
            return torch.quantile(flat_tensor, q)
        else:
            # Too large, sample uniformly
            logger.info(f"Tensor too large ({n_elements:,} elements), sampling {max_elements:,} for quantile")
            sample_size = max_elements
            indices = torch.randperm(n_elements, device='cpu')[:sample_size]
            sampled = flat_tensor.cpu()[indices]
            return torch.quantile(sampled, q).to(tensor.device)
    
    def _ties_dare_fusion(self, layer, W_vis: torch.Tensor, layer_idx: int):
        """Apply TIES (resolve conflicts) and DARE (add sparsity) to fusion"""

        # Get layer weights to modify
        target_weights = None
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            target_weights = layer.mlp.gate_proj.weight.data
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'fc_in'):
            target_weights = layer.mlp.fc_in.weight.data
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'up_proj'):
            target_weights = layer.mlp.up_proj.weight.data

        if target_weights is None:
            return  # Skip if no suitable MLP weights found

        # Ensure W_vis is on the same device and dtype
        W_vis = W_vis.to(target_weights.device).to(target_weights.dtype)

        # Handle dimension mismatch by projecting/slicing
        if W_vis.shape[0] < target_weights.shape[0] or W_vis.shape[1] < target_weights.shape[1]:
            # Pad W_vis if it's smaller
            pad_0 = max(0, target_weights.shape[0] - W_vis.shape[0])
            pad_1 = max(0, target_weights.shape[1] - W_vis.shape[1])
            if pad_0 > 0 or pad_1 > 0:
                W_vis_padded = torch.zeros(
                    target_weights.shape[0], target_weights.shape[1],
                    device=W_vis.device, dtype=W_vis.dtype
                )
                W_vis_padded[:W_vis.shape[0], :W_vis.shape[1]] = W_vis
                W_vis = W_vis_padded
        else:
            # Slice W_vis if it's larger
            W_vis = W_vis[:target_weights.shape[0], :target_weights.shape[1]]

        # Compute vision delta (simplified projection)
        beta = self.config.fusion_beta
        vis_delta = beta * (W_vis - target_weights)

        # TIES: Resolve sign conflicts by keeping larger magnitude
        current_sign = torch.sign(target_weights)
        delta_sign = torch.sign(vis_delta)
        conflict_mask = (current_sign != delta_sign) & (current_sign != 0)

        # Keep delta where it agrees or is larger magnitude
        ties_mask = conflict_mask & (torch.abs(vis_delta) >= torch.abs(target_weights))
        final_delta = torch.where(ties_mask, vis_delta, torch.zeros_like(vis_delta))

        # DARE: Drop small deltas and rescale survivors
        abs_delta = torch.abs(final_delta).float()  # Convert to float for quantile
        threshold = self._compute_quantile_safe(abs_delta, self.config.ties_drop_rate)
        drop_mask = torch.abs(final_delta) < threshold
        final_delta = torch.where(drop_mask, torch.zeros_like(final_delta),
                                final_delta * self.config.dare_rescale_factor)

        # Apply fusion
        target_weights.add_(final_delta)

    def _apply_language_delta(self, layer, param_path: str, delta: torch.Tensor):
        """
        Apply language knowledge delta to layer parameter using TIES+DARE

        Args:
            layer: The layer to modify
            param_path: Path to parameter (e.g., 'self_attn.q_proj', 'mlp.gate_proj')
            delta: The parameter delta to apply (already computed as large - small)
        """
        # Parse parameter path
        parts = param_path.split('.')
        target_obj = layer

        # Navigate to the target parameter
        for part in parts[:-1]:
            if not hasattr(target_obj, part):
                return  # Skip if path doesn't exist
            target_obj = getattr(target_obj, part)

        param_name = parts[-1]
        if not hasattr(target_obj, param_name):
            return

        target_param = getattr(target_obj, param_name)
        if not hasattr(target_param, 'weight'):
            return

        target_weights = target_param.weight.data
        device = target_weights.device
        dtype = target_weights.dtype

        # Move delta to correct device and dtype
        delta = delta.to(device).to(dtype)

        # Ensure shapes match (should already be aligned from _compute_weight_delta)
        if delta.shape != target_weights.shape:
            logger.warning(f"Shape mismatch for {param_path}: delta={delta.shape}, target={target_weights.shape}")
            return

        # Apply TIES: Resolve sign conflicts
        # Here delta represents knowledge from large model
        # We want to incorporate it, but resolve conflicts with current weights
        current_sign = torch.sign(target_weights)
        delta_sign = torch.sign(delta)

        # Conflict mask: where signs disagree
        conflict_mask = (current_sign != delta_sign) & (current_sign != 0) & (delta_sign != 0)

        # Keep delta where:
        # 1. No conflict (signs agree)
        # 2. Delta has larger magnitude (large model's knowledge is stronger)
        ties_mask = ~conflict_mask | (torch.abs(delta) >= torch.abs(target_weights))
        ties_delta = torch.where(ties_mask, delta, torch.zeros_like(delta))

        # Apply DARE: Drop small magnitude deltas and rescale
        # This prevents catastrophic forgetting by only keeping significant changes
        abs_delta = torch.abs(ties_delta).float()
        if abs_delta.numel() > 0 and abs_delta.max() > 0:
            threshold = self._compute_quantile_safe(abs_delta, self.config.ties_drop_rate)
            dare_mask = torch.abs(ties_delta) > threshold

            # Count kept elements and rescale
            kept_elements = dare_mask.sum().item()
            if kept_elements > 0:
                rescale_factor = ties_delta.numel() / kept_elements
                final_delta = torch.where(dare_mask,
                                        ties_delta * self.config.dare_rescale_factor,
                                        torch.zeros_like(ties_delta))
            else:
                final_delta = torch.zeros_like(ties_delta)
        else:
            final_delta = torch.zeros_like(ties_delta)

        # Apply language knowledge delta with scaling
        # Use language_transfer_beta to control how much large model knowledge to incorporate
        scaled_delta = self.config.language_transfer_beta * final_delta
        target_weights.add_(scaled_delta)

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