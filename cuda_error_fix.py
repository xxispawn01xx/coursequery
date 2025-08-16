#!/usr/bin/env python3
"""
CUDA Error Fix Script for RTX 3060 12GB
Fixes common CUDA device-side assert errors and memory issues.
"""

import os
import sys
import subprocess
import logging

def setup_cuda_environment():
    """Set up CUDA environment variables for stable operation."""
    
    # Set CUDA debugging flags
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Reduce memory fragmentation
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print(" CUDA environment configured for RTX 3060 12GB")
    print("Environment variables set:")
    print(f"  CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"  TORCH_USE_CUDA_DSA: {os.environ.get('TORCH_USE_CUDA_DSA')}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

def check_cuda_status():
    """Check CUDA installation and GPU status."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return torch.cuda.is_available()
    
    except ImportError:
        print(" PyTorch not installed")
        return False

def clear_cuda_cache():
    """Clear CUDA cache and reset GPU state."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(" CUDA cache cleared")
        else:
            print("ℹ️ CUDA not available - cache clear skipped")
    except Exception as e:
        print(f" Error clearing CUDA cache: {e}")

def run_memory_test():
    """Test GPU memory allocation."""
    try:
        import torch
        if not torch.cuda.is_available():
            print(" CUDA not available for memory test")
            return False
        
        print("🧪 Testing GPU memory allocation...")
        
        # Test small allocation
        test_tensor = torch.randn(1000, 1000, device='cuda')
        print(f" Small allocation successful: {test_tensor.shape}")
        
        # Test larger allocation (4GB equivalent)
        large_tensor = torch.randn(16384, 16384, device='cuda')
        print(f" Large allocation successful: {large_tensor.shape}")
        
        # Clean up
        del test_tensor, large_tensor
        torch.cuda.empty_cache()
        print(" Memory test passed")
        return True
        
    except Exception as e:
        print(f" Memory test failed: {e}")
        return False

def main():
    """Main function to set up CUDA environment and run tests."""
    print(" CUDA Error Fix for RTX 3060 12GB")
    print("=" * 50)
    
    # Set up environment
    setup_cuda_environment()
    print()
    
    # Check CUDA status
    print(" CUDA Status Check:")
    cuda_available = check_cuda_status()
    print()
    
    if cuda_available:
        # Clear cache
        print("🧹 Clearing CUDA cache:")
        clear_cuda_cache()
        print()
        
        # Run memory test
        print("🧪 Running memory test:")
        test_passed = run_memory_test()
        print()
        
        if test_passed:
            print(" CUDA setup complete and tested successfully!")
            print("\n You can now run the application:")
            print("   streamlit run app.py --server.port 5000")
        else:
            print(" Memory test failed. Try the following:")
            print("1. Restart your computer")
            print("2. Close other GPU-intensive applications")
            print("3. Update your NVIDIA drivers")
    else:
        print(" CUDA not available. Check your PyTorch installation.")

if __name__ == "__main__":
    main()