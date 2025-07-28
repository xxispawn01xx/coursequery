#!/usr/bin/env python3
"""
RTX 3060 Memory Diagnosis Tool
Diagnoses what's consuming GPU memory before model loading
"""

import torch
import gc
import os
import psutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_all_gpu_memory():
    """Aggressively clear all GPU memory."""
    logger.info("🧹 Performing aggressive GPU memory cleanup...")
    
    # Clear Python garbage
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to reset memory stats
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
    
    logger.info("✅ GPU memory cleanup completed")

def check_gpu_processes():
    """Check what processes are using GPU memory."""
    logger.info("🔍 Checking GPU processes...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            logger.warning("🚨 GPU processes found:")
            for line in result.stdout.strip().split('\n'):
                logger.warning(f"  Process: {line}")
        else:
            logger.info("✅ No GPU compute processes found")
            
    except FileNotFoundError:
        logger.warning("nvidia-smi not found - cannot check GPU processes")

def diagnose_rtx_3060_memory():
    """Comprehensive RTX 3060 memory diagnosis."""
    logger.info("🔬 RTX 3060 Memory Diagnosis Starting...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    # Get initial memory state
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info(f"📊 Initial RTX 3060 Memory State:")
    logger.info(f"  Total: {total:.2f} GB")
    logger.info(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    logger.info(f"  Reserved: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    logger.info(f"  Free: {total-reserved:.2f} GB")
    
    # Check if memory is already problematic
    if allocated > total * 0.8:
        logger.error(f"🚨 RTX 3060 memory already critically high: {allocated/total*100:.1f}%")
        return False
    
    # Check GPU processes
    check_gpu_processes()
    
    # Test small allocation
    try:
        logger.info("🧪 Testing small GPU allocation...")
        test_tensor = torch.ones(1000, 1000).cuda()
        logger.info("✅ Small allocation successful")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"❌ Small allocation failed: {e}")
        return False
    
    # Test medium allocation (1GB)
    try:
        logger.info("🧪 Testing 1GB GPU allocation...")
        # 1024*1024*256 float32 = 1GB
        test_tensor = torch.ones(1024, 1024, 256, dtype=torch.float32).cuda()
        logger.info("✅ 1GB allocation successful")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"❌ 1GB allocation failed: {e}")
        return False
    
    # Check memory after tests
    allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
    logger.info(f"📊 RTX 3060 memory after tests: {allocated_after:.2f} GB ({allocated_after/total*100:.1f}%)")
    
    if allocated_after < total * 0.1:
        logger.info("✅ RTX 3060 memory healthy - ready for model loading")
        return True
    else:
        logger.warning(f"⚠️ RTX 3060 memory elevated: {allocated_after/total*100:.1f}%")
        return False

def fix_memory_fragmentation():
    """Fix RTX 3060 memory fragmentation."""
    logger.info("🔧 Fixing RTX 3060 memory fragmentation...")
    
    # Set PyTorch memory allocation configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear everything
    clear_all_gpu_memory()
    
    logger.info("✅ Memory fragmentation fix applied")

if __name__ == "__main__":
    print("RTX 3060 Memory Diagnosis Tool")
    print("=" * 50)
    
    # Check initial state
    if not diagnose_rtx_3060_memory():
        print("\n🔧 Applying memory fixes...")
        fix_memory_fragmentation()
        
        print("\n🔄 Re-testing after fixes...")
        if diagnose_rtx_3060_memory():
            print("\n✅ RTX 3060 memory issues resolved!")
        else:
            print("\n❌ RTX 3060 memory issues persist - may need system restart")
    else:
        print("\n✅ RTX 3060 memory is healthy!")