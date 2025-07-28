"""
Configuration settings for the Real Estate AI Stack.
"""

import os
from pathlib import Path
from typing import Dict, Any

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
        # This is a fully local app - never skip model loading
        self.is_replit = False
        self.skip_model_loading = False
        
        self.model_config = {
            'mistral': {
                'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',  # Original Mistral model
                'max_length': 4096,
                'temperature': 0.7,
                'device': 'auto',  # Will use GPU if available
                'load_in_4bit': True,  # For RTX 3060 efficiency
            },
            'whisper': {
                'model_size': 'medium',  # Good balance of speed and accuracy
                'device': 'cuda' if self.has_gpu() else 'cpu',
            },
            'embeddings': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'device': 'cuda' if self.has_gpu() else 'cpu',
            }
        }
    
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
