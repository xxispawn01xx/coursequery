#!/usr/bin/env python3
"""
RTX 3060 Memory Diagnostic Tool
Comprehensive testing for faulty GPU memory issues
"""

import torch
import gc
import os
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTX3060MemoryTester:
    """Comprehensive RTX 3060 memory testing and fallback system."""
    
    def __init__(self):
        self.device_name = None
        self.total_memory = 0
        self.gpu_healthy = False
        
    def test_basic_gpu_access(self):
        """Test basic GPU accessibility."""
        logger.info("🔍 Testing basic GPU access...")
        
        try:
            if not torch.cuda.is_available():
                logger.error("❌ CUDA not available")
                return False
                
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"✅ GPU detected: {self.device_name}")
            logger.info(f"✅ Total memory: {self.total_memory:.1f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic GPU access failed: {e}")
            return False
    
    def test_memory_allocation(self):
        """Test GPU memory allocation in incremental steps."""
        logger.info("🔍 Testing GPU memory allocation...")
        
        try:
            # Test small allocation first
            logger.info("Testing 100MB allocation...")
            small_tensor = torch.zeros(25 * 1024 * 1024, dtype=torch.float32).cuda()  # ~100MB
            del small_tensor
            torch.cuda.empty_cache()
            logger.info("✅ Small allocation successful")
            
            # Test medium allocation
            logger.info("Testing 1GB allocation...")
            medium_tensor = torch.zeros(256 * 1024 * 1024, dtype=torch.float32).cuda()  # ~1GB
            del medium_tensor
            torch.cuda.empty_cache()
            logger.info("✅ Medium allocation successful")
            
            # Test large allocation
            logger.info("Testing 4GB allocation...")
            large_tensor = torch.zeros(1024 * 1024 * 1024, dtype=torch.float32).cuda()  # ~4GB
            del large_tensor
            torch.cuda.empty_cache()
            logger.info("✅ Large allocation successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory allocation failed: {e}")
            if "device-side assert" in str(e):
                logger.error("🚨 CONFIRMED: RTX 3060 has faulty memory (device-side assert)")
                logger.error("This is a hardware issue - GPU memory is corrupted")
            return False
    
    def test_compute_operations(self):
        """Test GPU compute operations for stability."""
        logger.info("🔍 Testing GPU compute operations...")
        
        try:
            # Matrix multiplication test
            logger.info("Testing matrix multiplication...")
            a = torch.randn(1000, 1000).cuda()
            b = torch.randn(1000, 1000).cuda()
            c = torch.matmul(a, b)
            del a, b, c
            torch.cuda.empty_cache()
            logger.info("✅ Matrix multiplication successful")
            
            # Iterative compute test
            logger.info("Testing iterative operations (stress test)...")
            for i in range(10):
                x = torch.randn(500, 500).cuda()
                y = x * x + x.sin() + x.cos()
                del x, y
                torch.cuda.empty_cache()
                time.sleep(0.1)  # Brief pause
            
            logger.info("✅ Iterative operations successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Compute operations failed: {e}")
            if "device-side assert" in str(e):
                logger.error("🚨 CONFIRMED: RTX 3060 compute units have issues")
            return False
    
    def test_model_loading_simulation(self):
        """Simulate model loading patterns that typically fail."""
        logger.info("🔍 Testing model loading simulation...")
        
        try:
            # Simulate embedding model loading
            logger.info("Simulating embedding model loading...")
            embedding_sim = torch.randn(30522, 384).cuda()  # BERT-like embeddings
            del embedding_sim
            torch.cuda.empty_cache()
            
            # Simulate transformer layers
            logger.info("Simulating transformer layers...")
            for layer in range(5):
                attention = torch.randn(12, 512, 512).cuda()  # Multi-head attention
                ffn = torch.randn(512, 2048).cuda()  # Feed-forward
                del attention, ffn
                torch.cuda.empty_cache()
            
            logger.info("✅ Model loading simulation successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading simulation failed: {e}")
            return False
    
    def comprehensive_test(self):
        """Run all tests and determine GPU health."""
        logger.info("🔬 Starting comprehensive RTX 3060 memory test...")
        logger.info("=" * 60)
        
        # Test 1: Basic access
        if not self.test_basic_gpu_access():
            logger.error("🚨 FATAL: Cannot access GPU at all")
            return False
        
        # Test 2: Memory allocation
        memory_ok = self.test_memory_allocation()
        
        # Test 3: Compute operations
        compute_ok = self.test_compute_operations()
        
        # Test 4: Model loading simulation
        model_ok = self.test_model_loading_simulation()
        
        # Final assessment
        self.gpu_healthy = memory_ok and compute_ok and model_ok
        
        logger.info("=" * 60)
        logger.info("🔬 DIAGNOSTIC RESULTS:")
        logger.info(f"GPU: {self.device_name}")
        logger.info(f"Memory: {self.total_memory:.1f}GB")
        logger.info(f"Memory allocation: {'✅ PASS' if memory_ok else '❌ FAIL'}")
        logger.info(f"Compute operations: {'✅ PASS' if compute_ok else '❌ FAIL'}")
        logger.info(f"Model loading sim: {'✅ PASS' if model_ok else '❌ FAIL'}")
        logger.info(f"Overall health: {'✅ HEALTHY' if self.gpu_healthy else '❌ FAULTY'}")
        
        if not self.gpu_healthy:
            logger.error("🚨 CONCLUSION: RTX 3060 has hardware memory issues")
            logger.error("💡 RECOMMENDATION: Use CPU-only mode for AI models")
            logger.error("🔧 SOLUTION: All models will automatically fall back to CPU")
        
        return self.gpu_healthy

def run_gpu_diagnostics():
    """Run the complete GPU diagnostic suite."""
    tester = RTX3060MemoryTester()
    
    # Set debug flags for better error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Run tests
    gpu_healthy = tester.comprehensive_test()
    
    # Write results to file for the main app
    results_file = Path("gpu_test_results.txt")
    with open(results_file, "w") as f:
        f.write(f"GPU_HEALTHY={gpu_healthy}\n")
        f.write(f"DEVICE_NAME={tester.device_name}\n")
        f.write(f"TOTAL_MEMORY={tester.total_memory:.1f}\n")
    
    logger.info(f"Results saved to {results_file}")
    return gpu_healthy

if __name__ == "__main__":
    success = run_gpu_diagnostics()
    sys.exit(0 if success else 1)