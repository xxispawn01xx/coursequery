#!/usr/bin/env python3
"""
GPU Memory Health Test for RTX 3060
Tests for memory corruption, stability, and performance issues
"""

import torch
import time
import os
import sys
import psutil
import gc
from datetime import datetime
import numpy as np

class GPUMemoryTester:
    def __init__(self):
        self.device = None
        self.total_memory = 0
        self.results = []
        
    def setup_cuda(self):
        """Initialize CUDA and get GPU info"""
        print("=== GPU Memory Health Test ===")
        print(f"Test started: {datetime.now()}")
        print()
        
        if not torch.cuda.is_available():
            print(" CUDA not available")
            return False
            
        self.device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        self.total_memory = torch.cuda.get_device_properties(0).total_memory
        
        print(f"🖥️  GPU: {gpu_name}")
        print(f" Total VRAM: {self.total_memory / (1024**3):.1f} GB")
        print(f" CUDA Version: {torch.version.cuda}")
        print(f"🐍 PyTorch Version: {torch.__version__}")
        print()
        
        return True
    
    def test_basic_allocation(self):
        """Test basic memory allocation patterns"""
        print(" Test 1: Basic Memory Allocation")
        
        try:
            # Start with small allocations
            for size_mb in [100, 500, 1000, 2000, 4000]:
                size_bytes = size_mb * 1024 * 1024
                if size_bytes > self.total_memory * 0.8:  # Don't exceed 80% of VRAM
                    break
                    
                print(f"   Allocating {size_mb}MB...", end='')
                tensor = torch.randn(size_bytes // 4, device=self.device, dtype=torch.float32)
                
                # Simple operations
                result = tensor.sum()
                tensor_copy = tensor.clone()
                
                print(f" Success (Sum: {result.item():.2e})")
                
                # Clean up
                del tensor, tensor_copy, result
                torch.cuda.empty_cache()
                time.sleep(0.1)
                
            self.results.append("Basic allocation: PASSED")
            return True
            
        except Exception as e:
            print(f" Failed: {e}")
            self.results.append(f"Basic allocation: FAILED - {e}")
            return False
    
    def test_memory_stress(self):
        """Stress test memory with repeated allocations"""
        print("\n🔥 Test 2: Memory Stress Test")
        
        try:
            max_safe_memory = int(self.total_memory * 0.7)  # Use 70% of VRAM
            chunk_size = max_safe_memory // 10  # Allocate in chunks
            
            tensors = []
            
            for i in range(10):
                print(f"   Stress allocation {i+1}/10 ({chunk_size/1024**2:.0f}MB)...", end='')
                
                # Allocate tensor
                tensor = torch.randn(chunk_size // 4, device=self.device, dtype=torch.float32)
                
                # Perform computation to test memory integrity
                tensor = tensor * 2.0 + 1.0
                checksum = tensor.sum().item()
                
                tensors.append((tensor, checksum))
                print(f" (checksum: {checksum:.2e})")
                
            # Verify all tensors still have correct checksums
            print("   Verifying memory integrity...", end='')
            for i, (tensor, original_checksum) in enumerate(tensors):
                current_checksum = tensor.sum().item()
                if abs(current_checksum - original_checksum) > 1e-3:
                    raise RuntimeError(f"Memory corruption detected in tensor {i}")
            
            print(" All checksums verified")
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            self.results.append("Memory stress: PASSED")
            return True
            
        except Exception as e:
            print(f" Failed: {e}")
            self.results.append(f"Memory stress: FAILED - {e}")
            torch.cuda.empty_cache()
            return False
    
    def test_embedding_simulation(self):
        """Simulate sentence transformer embedding workload"""
        print("\n🔤 Test 3: Embedding Workload Simulation")
        
        try:
            # Simulate sentence transformer dimensions
            batch_size = 32
            seq_length = 512
            hidden_dim = 384
            
            print(f"   Simulating batch processing ({batch_size} sequences)...", end='')
            
            # Create input tensors (simulating tokenized text)
            input_ids = torch.randint(0, 30000, (batch_size, seq_length), device=self.device)
            
            # Simulate embedding lookup
            embedding_weights = torch.randn(30000, hidden_dim, device=self.device)
            embeddings = torch.nn.functional.embedding(input_ids, embedding_weights)
            
            # Simulate transformer operations
            for layer in range(6):  # 6 layers like MiniLM
                # Attention simulation
                attention_weights = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
                embeddings = embeddings + attention_weights
                
                # Layer norm simulation
                embeddings = torch.nn.functional.layer_norm(embeddings, [hidden_dim])
                
                print(f".", end='')
            
            # Final pooling (mean)
            sentence_embeddings = embeddings.mean(dim=1)
            
            # Verify output
            assert sentence_embeddings.shape == (batch_size, hidden_dim)
            assert not torch.isnan(sentence_embeddings).any()
            
            print(f" Generated {batch_size} embeddings")
            
            # Clean up
            del input_ids, embedding_weights, embeddings, attention_weights, sentence_embeddings
            torch.cuda.empty_cache()
            
            self.results.append("Embedding simulation: PASSED")
            return True
            
        except Exception as e:
            print(f" Failed: {e}")
            self.results.append(f"Embedding simulation: FAILED - {e}")
            torch.cuda.empty_cache()
            return False
    
    def test_memory_fragmentation(self):
        """Test for memory fragmentation issues"""
        print("\n🧩 Test 4: Memory Fragmentation Test")
        
        try:
            tensors = []
            
            # Allocate many small tensors
            print("   Allocating 100 small tensors...", end='')
            for i in range(100):
                tensor = torch.randn(1024, 1024, device=self.device)  # 4MB each
                tensors.append(tensor)
                if i % 20 == 0:
                    print(".", end='')
            
            print(" ")
            
            # Free every other tensor (create fragmentation)
            print("   Creating fragmentation...", end='')
            for i in range(0, len(tensors), 2):
                del tensors[i]
                tensors[i] = None
            
            torch.cuda.empty_cache()
            print(" ")
            
            # Try to allocate a large tensor (should work if defragmentation works)
            print("   Testing large allocation after fragmentation...", end='')
            large_tensor = torch.randn(10 * 1024 * 1024, device=self.device)  # 40MB
            
            # Verify
            checksum = large_tensor.sum().item()
            print(f" (checksum: {checksum:.2e})")
            
            # Clean up
            del tensors, large_tensor
            torch.cuda.empty_cache()
            
            self.results.append("Memory fragmentation: PASSED")
            return True
            
        except Exception as e:
            print(f" Failed: {e}")
            self.results.append(f"Memory fragmentation: FAILED - {e}")
            torch.cuda.empty_cache()
            return False
    
    def test_temperature_stability(self):
        """Test GPU under sustained load to check for thermal issues"""
        print("\n🌡️  Test 5: Temperature Stability Test")
        
        try:
            # Run computation for 30 seconds
            print("   Running sustained computation for 30 seconds...")
            
            # Create workload that uses GPU compute units
            matrix_size = 1024
            a = torch.randn(matrix_size, matrix_size, device=self.device)
            b = torch.randn(matrix_size, matrix_size, device=self.device)
            
            start_time = time.time()
            iteration = 0
            
            while time.time() - start_time < 30:
                # Matrix multiplication (compute intensive)
                c = torch.matmul(a, b)
                
                # Add some randomness
                a = a + torch.randn_like(a) * 0.01
                b = b + torch.randn_like(b) * 0.01
                
                iteration += 1
                
                if iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"   {elapsed:.1f}s - iteration {iteration}")
            
            # Final verification
            final_result = torch.matmul(a, b).sum().item()
            print(f" Completed {iteration} iterations (final result: {final_result:.2e})")
            
            # Clean up
            del a, b, c
            torch.cuda.empty_cache()
            
            self.results.append("Temperature stability: PASSED")
            return True
            
        except Exception as e:
            print(f" Failed: {e}")
            self.results.append(f"Temperature stability: FAILED - {e}")
            torch.cuda.empty_cache()
            return False
    
    def print_system_info(self):
        """Print additional system information"""
        print("\n System Information:")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"   System RAM: {memory.total / (1024**3):.1f} GB")
        print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
        
        # CUDA memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            print(f"   CUDA Allocated: {allocated / (1024**2):.1f} MB")
            print(f"   CUDA Cached: {cached / (1024**2):.1f} MB")
        
        # Driver info
        print(f"   CUDA Driver: {torch.version.cuda}")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        if not self.setup_cuda():
            return False
        
        self.print_system_info()
        print()
        
        # Run all tests
        tests = [
            self.test_basic_allocation,
            self.test_memory_stress,
            self.test_embedding_simulation,
            self.test_memory_fragmentation,
            self.test_temperature_stability
        ]
        
        passed = 0
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f" Test failed with exception: {e}")
        
        # Print results
        print("\n" + "="*50)
        print("🏁 TEST RESULTS:")
        print("="*50)
        
        for result in self.results:
            status = " " if "PASSED" in result else " "
            print(f"{status} {result}")
        
        print(f"\nOverall: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("🎉 All tests passed! Your GPU memory appears healthy.")
        elif passed >= len(tests) * 0.8:
            print(" Most tests passed, but some issues detected. Monitor GPU carefully.")
        else:
            print("🚨 Multiple test failures. GPU may have hardware issues.")
            print("   Consider:\n   - Checking GPU temperature\n   - Testing in a different system\n   - Running manufacturer diagnostics")
        
        return passed == len(tests)

if __name__ == "__main__":
    tester = GPUMemoryTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()