"""
Local Model Manager for offline AI processing
Provides GPU-accelerated local models for embeddings and text generation
Designed for RTX 3060 and other CUDA-compatible GPUs
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Try importing required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU acceleration disabled")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available - local embeddings disabled")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - local text generation disabled")

class LocalModelManager:
    """Manages local AI models for offline operation"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize local model manager"""
        self.models_dir = models_dir or Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model instances
        self.embedding_model = None
        self.text_model = None
        self.device = self._get_device()
        
        logger.info(f"LocalModelManager initialized with device: {self.device}")
        
    def _get_device(self) -> str:
        """Determine the best available device"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> bool:
        """Load local embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available - cannot load embedding model")
            return False
            
        try:
            cache_dir = self.models_dir / "embeddings"
            cache_dir.mkdir(exist_ok=True)
            
            self.embedding_model = SentenceTransformer(
                model_name, 
                cache_folder=str(cache_dir),
                device=self.device
            )
            
            logger.info(f"Successfully loaded embedding model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def load_text_model(self, model_name: str = "microsoft/DialoGPT-small") -> bool:
        """Load local text generation model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available - cannot load text model")
            return False
            
        try:
            cache_dir = self.models_dir / "text_generation"
            cache_dir.mkdir(exist_ok=True)
            
            # Load tokenizer and model with caching
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir)
            )
            
            if self.device == "cuda":
                # GPU loading with memory optimization
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True if hasattr(torch, 'cuda') else False
                )
            else:
                # CPU loading
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir)
                )
                self.text_model.to(self.device)
            
            logger.info(f"Successfully loaded text model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            return False
    
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for texts"""
        if not self.embedding_model:
            logger.error("Embedding model not loaded")
            return None
            
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def generate_text(self, prompt: str, max_length: int = 512) -> Optional[str]:
        """Generate text using local model"""
        if not self.text_model:
            logger.error("Text model not loaded")
            return None
            
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt):].strip()  # Remove prompt from output
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'device': self.device,
            'torch_available': TORCH_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'embedding_model_loaded': self.embedding_model is not None,
            'text_model_loaded': self.text_model is not None,
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'models_dir': str(self.models_dir)
        }
    
    def unload_models(self):
        """Unload models to free memory"""
        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None
            
        if self.text_model:
            del self.text_model
            self.text_model = None
            
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Models unloaded and memory cleared")
    
    def is_available(self) -> bool:
        """Check if local models are available"""
        return TORCH_AVAILABLE and (SENTENCE_TRANSFORMERS_AVAILABLE or TRANSFORMERS_AVAILABLE)

# Create a default instance for easy importing
default_model_manager = LocalModelManager()

def get_default_model_manager() -> LocalModelManager:
    """Get the default model manager instance"""
    return default_model_manager