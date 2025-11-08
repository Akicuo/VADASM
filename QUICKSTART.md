# V-ADASM Quickstart

## ⚡ 5-Minute Setup

### Option 1: GPU Installation (Recommended)
```bash
# Clone and cd
git clone https://github.com/yourorg/vadasm.git
cd vadasm

# Install with GPU support
pip install -r requirements-gpu.txt

# Install V-ADASM
pip install -e .

# Verify
python -c "from vadasm import VADASMMerger; print('✅ V-ADASM ready!')"
```

### Option 2: CPU Installation
```bash
git clone https://github.com/yourorg/vadasm.git
cd vadasm

pip install -r requirements.txt
pip install -e .

python -c "from vadasm import VADASMMerger; print('✅ V-ADASM ready!')"
```

## 🚀 Quick Merge Demo

```bash
# Simple text-only merge (fast)
python scripts/vmerge.py --small distilgpt2 --large microsoft/DialoGPT-medium --no-vision --output ./demo-text-merged

# Full vision merge (2-4 hours depending on hardware)  
python scripts/vmerge.py --small microsoft/phi-2 --large llava-hf/llava-1.5-7b-hf --output ./demo-vlm-merged

# Interactive tutorial
jupyter notebook examples/vadasm_quickstart.ipynb
```

## 📋 System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU with ≥8GB VRAM (optional but recommended)
- **RAM**: 16GB+ 
- **Storage**: 20GB+ for model downloads

## 🧪 Test Your Installation

```bash
# Import test
python -c "from vadasm import VADASMMerger; print('✅ Good!')"

# CLI test  
python scripts/vmerge.py --help

# Notebook demo
jupyter notebook examples/vadasm_quickstart.ipynb
```

## 🐛 Common Issues

- **"torch" not found**: Run `pip install torch torchvision torchaudio` first
- **CUDA error**: Use CPU mode with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- **Memory issues**: Try smaller models like `distilgpt2` first
- **Permission denied**: Use virtual environment or install without `--user`

## 📈 Expected Performance

| Merge Type | Runtime | VCR (Vision) | TDP (Text Drop) | Params |
|------------|---------|--------------|-----------------|--------|
| Text-only | 15-30min | - | ±1% | Same |
| Vision-full | 2-4hrs | +15% | -2% | Same |

## 🎯 Next Steps

1. **Run the demo**: `jupyter notebook examples/vadasm_quickstart.ipynb`  
2. **Learn merging**: Read `docs/` for API reference
3. **Advanced usage**: See `scripts/` and `examples/`
4. **Contribute**: [GitHub Issues](https://github.com/yourorg/vadasm/issues)

---

**Ready to build edge Vision-Language Models? Let's get V-ADASMin'!** 🤖🖼️