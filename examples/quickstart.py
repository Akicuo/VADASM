import torch
import torch.nn as nn
from vadasm.merger import VADASMMerger, ModelConfig, MergeConfig

# Define models
small_config = ModelConfig(
    name_or_path="microsoft/phi-2",  # Small text model (2.7B)
    is_moe=False,
    has_vision=False
)

large_config = ModelConfig(
    name_or_path="llava-hf/llava-1.5-7b-hf",  # Large multimodal model
    is_moe=False, 
    has_vision=True
)

# Configure merge
merge_config = MergeConfig(
    fusion_beta=0.3,  # Vision injection strength
    ties_drop_rate=0.3,  # TIES sparsification
    evo_generations=10  # Quick evolution
)

# Initialize merger
merger = VADASMMerger(merge_config)

# Merge models (training-free!)
merged_model = merger.merge_models(small_config, large_config)

print(f"Merged model has vision: {merged_model.config.has_vision}")
# Output: True - now processes image+text inputs!