#!/usr/bin/env python3
"""
RTX 3060 HuggingFace Forum Fix
Implementing exact solutions from GitHub issues #28284 and #22546
"""

import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_rtx_3060_forum_fixes():
    """Apply exact RTX 3060 fixes from HuggingFace forums."""
    logger.info(" Applying RTX 3060 forum fixes...")
    
    # Fix 1: Set RTX 3060 compute capability (8.6 for Ampere)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    
    # Fix 2: Disable accelerate auto device mapping that causes issues
    os.environ['ACCELERATE_USE_MPS_DEVICE'] = 'false'
    
    # Fix 3: Single GPU enforcement for RTX 3060
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Fix 4: Remove debug flags that can cause device-side assert
    os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
    os.environ.pop('TORCH_USE_CUDA_DSA', None)
    
    logger.info(" RTX 3060 environment configured")

def test_simple_model_loading():
    """Test simple model loading without complex device mapping."""
    apply_rtx_3060_forum_fixes()
    
    try:
        logger.info(" Testing simple model loading (forum method)...")
        
        # Use small model first
        model_name = "gpt2"
        
        # Load tokenizer (always safe)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Simple model loading without device_map (let PyTorch handle it)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Move to GPU manually (safer than device_map)
        if torch.cuda.is_available():
            logger.info("Moving model to GPU manually...")
            model = model.cuda()
        
        # Test inference
        logger.info("Testing inference...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=50
        )
        
        result = pipe("Hello, world!", max_length=20, do_sample=False)
        logger.info(f" Success: {result[0]['generated_text']}")
        
        # Cleanup
        del model, tokenizer, pipe
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f" Forum fix failed: {e}")
        if "device-side assert" in str(e):
            logger.error("Device-side assert still occurring - may need driver update")
        return False

if __name__ == "__main__":
    success = test_simple_model_loading()
    if success:
        logger.info("🎉 RTX 3060 forum fixes working!")
    else:
        logger.error("Forum fixes unsuccessful - may need different approach")