"""
Configuration settings for the Real Estate AI Stack.
"""

import os
from pathlib import Path
from typing import Dict, Any

# RTX 3060 Memory Fragmentation Fix - Apply BEFORE any imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Additional RTX 3060 stability fixes
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config:
    """Application configuration."""
    
    def __init__(self):
        """Initialize configuration."""
        self.base_dir = Path(__file__).parent
        self.setup_directories()
        self.setup_model_config()
    
    def setup_directories(self):
        """Create necessary directories."""
        self.raw_docs_dir = self.base_dir / "raw_docs"
        self.indexed_courses_dir = self.base_dir / "indexed_courses"
        self.models_dir = self.base_dir / "models"
        self.temp_dir = self.base_dir / "temp"
        
        # Create directories
        for directory in [self.raw_docs_dir, self.indexed_courses_dir, 
                         self.models_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
        
        # Create GGUF models subdirectory
        gguf_dir = self.models_dir / "gguf"
        gguf_dir.mkdir(exist_ok=True)
    
    def setup_model_config(self):
        """Configure model settings."""
        # Auto-detect environment - enable all models for local usage
        self.is_replit = os.environ.get('REPL_ID') is not None
        # Enable full AI functionality - user wants all models available
        self.skip_model_loading = False
        
        self.model_config = {
            'mistral': {
                'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',  # Perfect for RTX 3060 12GB
                'max_length': 4096,
                'temperature': 0.7,
                'device': 'cuda',  # Use GPU with 12GB VRAM
                'load_in_4bit': True,  # 4-bit quantization for efficiency
            },
            'llama': {
                'model_name': 'meta-llama/Llama-2-7b-chat-hf',  # Alternative for 12GB GPU
                'max_length': 4096,
                'temperature': 0.7,
                'device': 'cuda',
                'load_in_4bit': True,
            },
            'small': {
                'model_name': 'microsoft/DialoGPT-small',  # Fallback for system RAM issues
                'max_length': 512,
                'temperature': 0.7,
                'device': 'cpu',
                'load_in_4bit': False,
            },
            'medium': {
                'model_name': 'microsoft/DialoGPT-medium', 
                'max_length': 1024,
                'temperature': 0.7,
                'device': 'cuda',  # Can use GPU
                'load_in_4bit': False,
            },
            'whisper': {
                'model_size': 'medium',  # Good balance, can use GPU
                'device': 'cuda',  # RTX 3060 handles Whisper well
            },
            'embeddings': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'device': 'cuda',  # Use GPU for faster embeddings
            }
        }
        
        # With RTX 3060 12GB, prefer the full models
        self.preferred_model = 'mistral'  # RTX 3060 can handle this easily
    
    def _detect_replit(self) -> bool:
        """Detect if running on Replit environment."""
        # Check for Replit-specific environment variables
        replit_indicators = [
            'REPL_ID',
            'REPL_SLUG', 
            'REPLIT_DB_URL',
            'REPL_OWNER'
        ]
        
        for indicator in replit_indicators:
            if os.getenv(indicator):
                return True
        
        # Check if running in /home/runner (Replit path)
        if str(self.base_dir).startswith('/home/runner'):
            return True
            
        return False
    
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @property
    def chunk_config(self) -> Dict[str, Any]:
        """Configuration for document chunking."""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'syllabus_weight': 2.0,  # Syllabus content gets double weight
        }
    
    @property
    def index_config(self) -> Dict[str, Any]:
        """Configuration for indexing."""
        return {
            'similarity_top_k': 5,
            'response_mode': 'compact',
            'streaming': False,
        }
