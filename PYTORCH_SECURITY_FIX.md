# PyTorch Security Fix for CVE-2025-32434

## Security Issue
PyTorch versions < 2.6.0 have a critical vulnerability in `torch.load()` even with `weights_only=True`.
**CVE-2025-32434**: https://nvd.nist.gov/vuln/detail/CVE-2025-32434

## Local Installation Fix

For your RTX 3060 system, update PyTorch to version 2.6.0+ with CUDA support:

### 1. Uninstall Old PyTorch
```bash
pip uninstall torch torchvision torchaudio
```

### 2. Install Secure PyTorch with CUDA 12.1
```bash
pip install torch>=2.6.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install safetensors
```bash
pip install safetensors>=0.4.0
```

### 4. Run Installation Script
```bash
python install_dependencies.py
```

## Code Changes Made

The following security enhancements have been implemented:

1. **Force Safetensors**: All model loading now uses `use_safetensors=True`
2. **Disable Remote Code**: Set `trust_remote_code=False` for security
3. **Updated Dependencies**: `install_dependencies.py` now installs PyTorch 2.6.0+
4. **CUDA Environment**: Added CUDA stability settings for RTX 3060

## Model Loading Security

Models are now loaded with:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_safetensors=True,        # Avoid vulnerable torch.load
    trust_remote_code=False,     # Security best practice
    # ... other parameters
)
```

## Status

- **Replit**: Keeps CPU-only PyTorch (no security risk with disabled models)
- **Local**: Update to PyTorch 2.6.0+ with CUDA support required
- **Models**: All major models (Llama 2, Mistral) support safetensors format

## Next Steps

1. On your local RTX 3060 system, run the installation script
2. Verify PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Should be 2.6.0 or higher with `+cu121` for CUDA support
4. Test model loading with the updated security settings

The app will now use safetensors exclusively, avoiding the CVE-2025-32434 vulnerability.