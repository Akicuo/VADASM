# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

V-ADASM (Vision-Adaptive Dimensionality-Aligned Subspace Merging) is a training-free framework for merging large multimodal (text+vision) LLMs into compact text-only LLMs to create lightweight Vision-Language Models (VLMs). The system uses offline operations (SVD, permutations, evolutionary tuning) to inject visual reasoning from large donor models (e.g., LLaVA-Next 34B) into small base models (e.g., Llama-3-7B) without fine-tuning or distillation.

## Build & Development Commands

### Installation
```bash
# GPU installation (recommended)
pip install -r requirements-gpu.txt
pip install -e .

# CPU installation
pip install -r requirements.txt
pip install -e .

# Development dependencies
pip install -r requirements-dev.txt
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_merging.py

# Run with verbose output
pytest -v tests/
```

### Linting
```bash
# Lint with ruff (configured in pyproject.toml)
ruff check vadasm/
ruff check scripts/

# Auto-fix issues
ruff check --fix vadasm/
```

### Running Merges
```bash
# Simple text-only merge (fast demo)
python scripts/vmerge.py --small distilgpt2 --large microsoft/DialoGPT-medium --no-vision --output ./demo-text-merged

# Full vision merge (2-4 hours)
python scripts/vmerge.py --small meta-llama/Llama-3-7B --large llava-hf/llava-v1.6-7b --output ./vadasm-7b

# With validation data
python scripts/vmerge.py --small microsoft/phi-2 --large llava-hf/llava-1.5-7b-hf --output ./demo-vlm --val_vision laion/coco
```

### Evaluation
```bash
# Evaluate merged model
python scripts/eval_vlm.py --model ./vadasm-7b --tasks vqav2 mmlu

# Multiple tasks
python scripts/eval_vlm.py --model ./vadasm-7b --tasks vqav2 okvqa mmlu gsm8k --limit 1000
```

## Architecture Overview

### Core Pipeline (5-Step V-ADASM Process)

The merger follows a 5-step pipeline implemented in `vadasm/merger.py`:

1. **Vision Subspace Extraction** (`VisionExtractor`, `_extract_vision_subspaces`): Extracts vision capabilities from large donor model using SVD decomposition on the multimodal projector, reducing to principal components capturing 95% variance.

2. **Cross-Modality Alignment** (`CrossModalAligner`, `_cross_modality_alignment`): Aligns text and vision representations using Hungarian algorithm to find optimal neuron permutations based on cosine similarity. Aligns the first 20% of layers where vision-text crossover occurs.

3. **Subspace Fusion & Injection** (`SubspaceFuser`, `_subspace_fusion_injection`): Injects vision subspaces into small model using TIES (resolves sign conflicts, keeps larger magnitudes) and DARE (drops 30% of small magnitude deltas, rescales survivors).

4. **Evolutionary Hyperparameter Optimization** (`EvolutionaryTuner`, `_evolutionary_tuning`): Uses DEAP genetic algorithm to optimize fusion_beta and ties_drop_rate parameters on validation data (30 population, 15 generations).

5. **Validation & Deployment** (`_validate_merge`): Final parameter counting, vision/text scoring, and model export with vision_projector module attached.

### Module Responsibilities

- **`vadasm/merger.py`**: Main `VADASMMerger` class orchestrating the 5-step pipeline. Contains `ModelConfig` and `MergeConfig` dataclasses for configuration.

- **`vadasm/vision.py`**: `VisionExtractor` class handles vision tower and projector extraction, SVD decomposition, and cross-modal activation generation.

- **`vadasm/alignment.py`**: `CrossModalAligner` implements Hungarian algorithm for neuron alignment, handles dimension mismatches between dense/MoE architectures.

- **`vadasm/fusion.py`**: `SubspaceFuser` applies TIES+DARE fusion algorithms, injects vision_projector into final model, updates model config with vision capability flags.

- **`vadasm/evolution.py`**: `EvolutionaryTuner` wraps DEAP for hyperparameter optimization using genetic algorithms.

- **`vadasm/utils.py`**: Standalone utility functions (`ties_merge`, `dare_merge`, `hungarian_neuron_alignment`, `svd_subspace_reduction`) that can be used independently.

### MoE (Mixture-of-Experts) Support

The architecture handles heterogeneous Dense ↔ MoE merges:
- MoE layers aggregate over top-k experts (default k=2) using `moe_aggregate_method` (norm_avg or uniform)
- `_get_layer_activations_moe` averages expert outputs for alignment
- Router probabilities used to weight expert contributions during fusion

### Key Configuration Parameters

- **`projector_svd_rank`** (0.95): Variance threshold for SVD - controls how much vision information is retained
- **`alignment_layer_ratio`** (0.2): Fraction of layers to align from start - where vision-text crossover happens
- **`fusion_beta`** (0.3): Weight for vision deltas during fusion - higher = more vision influence
- **`ties_drop_rate`** (0.3): Fraction of deltas to drop in TIES - controls sparsity
- **`evo_generations`** (15): Number of evolutionary optimization rounds

## Important Implementation Details

### Weight Loading & Device Management
- Models are loaded with `torch_dtype` specified in config (default: bfloat16)
- Device auto-detection: uses CUDA if available, falls back to CPU
- Multimodal models return dict with "model" + "processor", text-only returns "model" + "tokenizer"

### Vision Projector Injection
- The final merged model has a `vision_projector` attribute (nn.Linear layer) for multimodal inference
- Model config updated with `has_vision=True` and `vision_config` dict containing projector dimensions
- Output structure in `vadasm_config.json` includes merge method, source models, and vision capability flags

### Alignment Permutations
- Permutations are applied to attention weights (q_proj, k_proj, v_proj) to align neuron correspondences
- Only affects early layers (first 20% by default) where modalities interact
- Uses cosine similarity cost matrix with Hungarian algorithm for optimal assignment

### TIES + DARE Fusion Logic
1. Compute delta: `beta * (vision_weights - base_weights)`
2. TIES resolves conflicts: keep delta where it agrees with base or has larger magnitude
3. DARE drops deltas below quantile threshold and rescales survivors to preserve total magnitude
4. Applied to MLP gate_proj or attention q_proj weights depending on layer structure

## Testing Strategy

Tests in `tests/test_merging.py` focus on:
- `test_ties_merge`: Verifies sparsification and shape preservation
- `test_dare_merge`: Checks sparsity increases after drop/rescale
- `test_hungarian_alignment`: Validates permutation indices are valid

When adding features, ensure alignment/fusion utilities remain pure functions testable in isolation.

## Known Limitations & Edge Cases

- Vision extraction requires models with `vision_tower` and `multi_modal_projector` attributes
- Dimension mismatches between small/large models handled via bilinear interpolation (simplified - production should use learned projectors)
- Evolutionary tuning requires DEAP library - gracefully falls back to default params if unavailable
- Cross-modal activations currently use synthetic data (placeholder) - production needs real image-text pair datasets
- Evaluation functions (`_evaluate_vision`, `_evaluate_text`) return dummy scores - integrate actual benchmarks for production

## Dependencies

Core requirements:
- `torch>=2.0.0` - Model operations
- `transformers>=4.35.0` - HuggingFace models
- `scipy>=1.10.0` - Linear sum assignment (Hungarian algorithm)
- `scikit-learn>=1.3.0` - Cosine similarity
- `numpy>=1.24.0` - Array operations
- `deap>=1.4.0` - Evolutionary optimization (optional)
- `datasets>=2.14.0` - Validation data loading
- `accelerate>=0.25.0` - Multi-GPU support

Additional for GPU: `torch` with CUDA support (see `requirements-gpu.txt`)

## File Structure Notes

- `scripts/vmerge.py`: CLI entry point for merging, handles argument parsing and model config creation
- `scripts/eval_vlm.py`: Evaluation CLI with task configs for VQAv2, OK-VQA, MMLU, GSM8K, etc.
- `examples/`: Jupyter notebooks for interactive demos (`vadasm_quickstart.ipynb`)
- `docs/`: Additional documentation (benchmarks.md, api.md)
