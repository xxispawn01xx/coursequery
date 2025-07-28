# PyTorch CUDA Setup for RTX 3060 12GB

## Quick Installation

### Option 1: Conda (Recommended)
```bash
# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Option 2: Pip
```bash
# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if 12.1 doesn't work)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verify Installation

Run this Python code to test:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output for RTX 3060:
```
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU device: NVIDIA GeForce RTX 3060
```

## Prerequisites

### 1. Install NVIDIA Drivers
- Download latest drivers from: https://www.nvidia.com/drivers
- Your RTX 3060 needs driver version 450+ for CUDA support

### 2. Install CUDA Toolkit (Optional but recommended)
- Download from: https://developer.nvidia.com/cuda-downloads
- Choose CUDA 12.1 for best PyTorch compatibility

### 3. Verify NVIDIA Installation
```bash
nvidia-smi
```
Should show your RTX 3060 and driver version.

## Troubleshooting

### If torch.cuda.is_available() returns False:

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Reinstall PyTorch with specific CUDA version:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Check CUDA installation:**
   ```bash
   nvcc --version
   ```

4. **Environment variables (if needed):**
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$PATH:$CUDA_HOME/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
   ```

## Performance Benefits with RTX 3060

With proper CUDA setup, you'll see:
- **5-10x faster** model loading
- **3-5x faster** inference
- **2-3x faster** embeddings
- **Lower CPU usage** (GPU handles AI tasks)

Your 12GB VRAM is perfect for:
- Mistral 7B (4-bit quantization): ~4GB VRAM
- Llama 2 7B (4-bit quantization): ~4GB VRAM  
- Embeddings: ~1GB VRAM
- **Total: ~9GB used, 3GB free for batching**