"""
V-ADASM CLI and helper functions
"""

import argparse
import torch
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from vadasm.merger import VADASMMerger, ModelConfig, MergeConfig
from vadasm.vision import VisionExtractor
from vadasm.alignment import CrossModalAligner  
from vadasm.fusion import SubspaceFuser
from vadasm.evolution import EvolutionaryTuner
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="V-ADASM Model Merger")
    
    # Model paths
    parser.add_argument("--small", type=str, required=True,
                       help="Path/name of small base model (recipient)")
    parser.add_argument("--large", type=str, required=True,
                       help="Path/name of large multimodal donor model (source)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for merged model")
    
    # Vision options
    parser.add_argument("--no_vision", action="store_true",
                       help="Skip multimodal features, do text-only ADASM")
    
    # Validation data
    parser.add_argument("--val_text", type=str, default=None,
                       help="Path to text validation data (JSON)")
    parser.add_argument("--val_vision", type=str, default=None, 
                       help="Path to vision validation data (JSON)")
    
    # Architecture configs
    parser.add_argument("--small_moe", action="store_true",
                       help="Small model is MoE")
    parser.add_argument("--large_moe", action="store_true",
                       help="Large model is MoE")
    parser.add_argument("--moe_top_k", type=int, default=2,
                       help="Top-k experts to aggregate in MoE")
    parser.add_argument("--moe_aggregate", type=str, default="norm_avg",
                       choices=["norm_avg", "uniform"], help="MoE aggregation method")
    
    # Hyperparameters
    parser.add_argument("--fusion_beta", type=float, default=0.3,
                       help="Fusion weight for vision deltas")
    parser.add_argument("--svd_rank", type=float, default=0.95,
                       help="SVD variance threshold")
    parser.add_argument("--ties_drop", type=float, default=0.3,
                       help="TIES drop rate")
    parser.add_argument("--evo_gens", type=int, default=15,
                       help="Evolutionary generations")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: cuda/cpu/auto")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["fp16", "bf16", "fp32"], help="Torch dtype")
    
    return parser.parse_args()

def create_model_configs(args) -> tuple[ModelConfig, ModelConfig]:
    """Create model configuration objects from args"""
    
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    
    small_config = ModelConfig(
        name_or_path=args.small,
        is_moe=args.small_moe,
        has_vision=False,  # Small models typically text-only
        vision_config=None
    )
    
    # Auto-detect vision capability for large model
    try:
        model = AutoModelForCausalLM.from_pretrained(args.large, torch_dtype=dtype_map[args.dtype])
        has_vision = hasattr(model, 'vision_tower') or hasattr(model, 'multi_modal_projector')
    except:
        has_vision = not args.no_vision  # Fallback to argument
    
    large_config = ModelConfig(
        name_or_path=args.large,
        is_moe=args.large_moe,
        has_vision=has_vision,
        vision_config={"top_k": args.moe_top_k, "aggregate": args.moe_aggregate} if args.large_moe else {}
    )
    
    return small_config, large_config

def load_validation_data(val_text_path, val_vision_path):
    """Load validation datasets"""
    val_data = {}
    
    if val_text_path and Path(val_text_path).exists():
        with open(val_text_path) as f:
            val_data["text"] = json.load(f)
        logger.info(f"Loaded {len(val_data['text'])} text validation samples")
    
    if val_vision_path and Path(val_vision_path).exists():
        with open(val_vision_path) as f:
            val_data["vision"] = json.load(f)
        logger.info(f"Loaded {len(val_data['vision'])} vision validation samples")
    
    return val_data if val_data else None

def save_merged_model(model, tokenizer_or_processor, output_path, config):
    """Save the merged model and configuration"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_path)
    logger.info(f"Model saved to {output_path}")
    
    # Save tokenizer/processor
    if isinstance(tokenizer_or_processor, dict) and "processor" in tokenizer_or_processor:
        tokenizer_or_processor["processor"].save_pretrained(output_path)
    else:
        tokenizer_or_processor.save_pretrained(output_path)
    
    # Save V-ADASM config for inference
    config_dict = {
        "merge_method": "V-ADASM" if config.large_config.has_vision else "ADASM",
        "small_model": config.small_config.name_or_path,
        "large_model": config.large_config.name_or_path,
        "has_vision": model.config.has_vision if hasattr(model, 'config') else False,
        "vision_config": model.config.vision_config if hasattr(model, 'config') and hasattr(model.config, 'vision_config') else {}
    }
    
    with open(output_path / "vadasm_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"V-ADASM config saved to {output_path}/vadasm_config.json")

def main():
    """Main V-ADASM merging function"""
    args = parse_args()
    
    # Setup configuration
    small_config, large_config = create_model_configs(args)
    
    merge_config = MergeConfig(
        projector_svd_rank=args.svd_rank,
        fusion_beta=args.fusion_beta,
        ties_drop_rate=args.ties_drop,
        torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype],
        device=args.device,
        evo_generations=args.evo_gens,
        moe_top_k=args.moe_top_k,
        moe_aggregate_method=args.moe_aggregate
    )
    
    # Load validation data
    val_data = load_validation_data(args.val_text, args.val_vision)
    
    logger.info("=== V-ADASM Model Merging ===")
    logger.info(f"Small model: {small_config.name_or_path}")
    logger.info(f"Large model: {large_config.name_or_path}")
    logger.info(f"Multimodal: {large_config.has_vision}")
    logger.info(f"Output: {args.output}")
    
    # Initialize merger
    merger = VADASMMerger(merge_config)
    
    # Perform merge
    merged_model = merger.merge_models(small_config, large_config, val_data)
    
    # Save result
    # Get tokenizer/processor from load (simplified - would need proper handling)
    if large_config.has_vision:
        from transformers import AutoProcessor
        tokenizer_or_processor = AutoProcessor.from_pretrained(large_config.name_or_path)
    else:
        tokenizer_or_processor = AutoTokenizer.from_pretrained(large_config.name_or_path)
    
    save_merged_model(merged_model, tokenizer_or_processor, args.output, merge_config)
    logger.info("✅ V-ADASM merge completed successfully!")

if __name__ == "__main__":
    main()