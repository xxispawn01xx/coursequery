#!/usr/bin/env python3
"""
Memory optimization utilities for local model loading.
Helps reduce memory usage when loading AI models locally.
"""

import gc
import logging
import os
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Utilities for optimizing memory usage during model loading."""
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "total_ram": memory.total / (1024**3),  # GB
                "available_ram": memory.available / (1024**3),  # GB
                "used_ram": memory.used / (1024**3),  # GB
                "ram_percent": memory.percent,
                "total_swap": swap.total / (1024**3),  # GB
                "used_swap": swap.used / (1024**3),  # GB
                "swap_percent": swap.percent
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return None
    
    @staticmethod
    def check_memory_available(required_gb: float = 4.0) -> bool:
        """Check if enough memory is available for model loading."""
        info = MemoryOptimizer.get_memory_info()
        if not info:
            return True  # Assume OK if we can't check
        
        available = info["available_ram"]
        logger.info(f"Available RAM: {available:.1f}GB, Required: {required_gb}GB")
        
        if available < required_gb:
            logger.warning(f"Low memory warning: {available:.1f}GB available, {required_gb}GB recommended")
            return False
        
        return True
    
    @staticmethod
    def clear_memory():
        """Force garbage collection to free memory."""
        logger.info("Clearing memory...")
        gc.collect()
        
        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
        except ImportError:
            pass
    
    @staticmethod
    def suggest_optimal_model():
        """Suggest the best model based on available memory."""
        info = MemoryOptimizer.get_memory_info()
        if not info:
            return "microsoft/DialoGPT-small"  # Safe fallback
        
        available = info["available_ram"]
        
        if available >= 8.0:
            return "mistralai/Mistral-7B-Instruct-v0.1"
        elif available >= 2.0:
            return "microsoft/DialoGPT-medium"
        else:
            return "microsoft/DialoGPT-small"
    
    @staticmethod
    def log_memory_status():
        """Log current memory status for debugging."""
        info = MemoryOptimizer.get_memory_info()
        if info:
            logger.info(f"Memory Status:")
            logger.info(f"  RAM: {info['used_ram']:.1f}/{info['total_ram']:.1f}GB ({info['ram_percent']:.1f}%)")
            logger.info(f"  Swap: {info['used_swap']:.1f}/{info['total_swap']:.1f}GB ({info['swap_percent']:.1f}%)")
            logger.info(f"  Available: {info['available_ram']:.1f}GB")
        
    @staticmethod
    def increase_virtual_memory_suggestions():
        """Provide suggestions for increasing virtual memory on Windows."""
        return """
💡 To fix memory issues on Windows:

1. **Increase Virtual Memory (Paging File):**
   - Press Windows + R, type "sysdm.cpl"
   - Go to Advanced tab → Performance Settings → Advanced → Virtual Memory
   - Click "Change" → Uncheck "Automatically manage"
   - Select your drive → Custom size
   - Set Initial: 4096 MB, Maximum: 8192 MB (or higher)
   - Click Set → OK → Restart computer

2. **Free up RAM:**
   - Close unnecessary programs
   - Use Task Manager to end memory-heavy processes
   - Restart your computer to clear memory

3. **Use smaller models:**
   - DialoGPT-small (~120MB) instead of Mistral-7B (~14GB)
   - Reduce model precision to save memory

4. **Alternative solutions:**
   - Use cloud-based models with API keys
   - Run models on a more powerful machine
   - Use model quantization to reduce memory usage
"""