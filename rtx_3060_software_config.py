#!/usr/bin/env python3
"""
RTX 3060 Software Configuration Tool
Since memtestcl passed (0 errors), the issue is software configuration, not hardware
"""

import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_minimal_model_loading():
    """Test minimal model loading to isolate the software issue."""
    logger.info(" Testing minimal model loading on healthy RTX 3060...")
    
    # Conservative environment setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Remove debug flags that might cause issues
    os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
    os.environ.pop('TORCH_USE_CUDA_DSA', None)
    
    try:
        # Test 1: Basic CUDA operations
        logger.info("Step 1: Testing basic CUDA operations...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Test 2: Small tensor operations
        logger.info("Step 2: Testing tensor operations...")
        x = torch.randn(10, 10).to(device)
        y = x @ x.T
        logger.info(f" Tensor ops successful: {y.shape}")
        
        # Test 3: Try loading a tiny model
        logger.info("Step 3: Testing tiny model loading...")
        model_name = "gpt2"  # Small, well-tested model
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try loading model with conservative settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map="auto",  # Let transformers handle device mapping
            low_cpu_mem_usage=True,
            trust_remote_code=False
        )
        
        logger.info(f" Model loaded successfully: {model.config.model_type}")
        
        # Test 4: Try inference
        logger.info("Step 4: Testing inference...")
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f" Inference successful: {result}")
        
        # Clean up
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        
        logger.info("🎉 All tests passed! RTX 3060 software config is working")
        return True
        
    except Exception as e:
        logger.error(f" Test failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        if "device-side assert" in str(e):
            logger.error("Device-side assert suggests software configuration issue")
            logger.error("This is NOT hardware failure (memtest passed)")
            
            # Try diagnostics
            logger.info(" Trying alternative configurations...")
            
            # Alternative 1: Force CPU offloading
            try:
                logger.info("Alternative 1: CPU offloading...")
                model = AutoModelForCausalLM.from_pretrained(
                    "gpt2",
                    device_map="cpu",
                    torch_dtype=torch.float32
                )
                logger.info(" CPU loading works - issue is GPU config")
                del model
                return False
            except Exception as e2:
                logger.error(f"CPU loading also failed: {e2}")
                
        return False

def suggest_fixes():
    """Suggest potential fixes for the software configuration."""
    logger.info(" Suggested fixes for RTX 3060 software configuration:")
    logger.info("1. Update PyTorch to latest version")
    logger.info("2. Update transformers library")
    logger.info("3. Clear CUDA cache completely")
    logger.info("4. Reset GPU driver state")
    logger.info("5. Use specific device mapping instead of 'auto'")
    
if __name__ == "__main__":
    success = test_minimal_model_loading()
    if not success:
        suggest_fixes()