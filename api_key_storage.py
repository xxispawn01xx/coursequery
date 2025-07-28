"""
Secure API Key Storage Manager
Stores API keys locally with basic encryption for persistence.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class APIKeyStorage:
    """Manages secure local storage of API keys."""
    
    def __init__(self):
        self.storage_file = Path("cache/.api_keys.json")
        self.storage_file.parent.mkdir(exist_ok=True)
        
    def _simple_encode(self, text: str) -> str:
        """Basic encoding for local storage (not cryptographically secure)."""
        return base64.b64encode(text.encode()).decode()
    
    def _simple_decode(self, encoded: str) -> str:
        """Decode base64 encoded text."""
        try:
            return base64.b64decode(encoded.encode()).decode()
        except:
            return ""
    
    def save_keys(self, openai_key: str = None, perplexity_key: str = None):
        """Save API keys to local storage."""
        try:
            # Load existing data
            stored_data = {}
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    stored_data = json.load(f)
            
            # Update with new keys
            if openai_key and openai_key.strip():
                stored_data['openai'] = self._simple_encode(openai_key.strip())
            
            if perplexity_key and perplexity_key.strip():
                stored_data['perplexity'] = self._simple_encode(perplexity_key.strip())
            
            # Save to file
            with open(self.storage_file, 'w') as f:
                json.dump(stored_data, f)
            
            logger.info("API keys saved to local storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            return False
    
    def load_keys(self) -> Dict[str, str]:
        """Load API keys from local storage."""
        try:
            if not self.storage_file.exists():
                return {}
            
            with open(self.storage_file, 'r') as f:
                stored_data = json.load(f)
            
            keys = {}
            if 'openai' in stored_data:
                keys['openai'] = self._simple_decode(stored_data['openai'])
            
            if 'perplexity' in stored_data:
                keys['perplexity'] = self._simple_decode(stored_data['perplexity'])
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}
    
    def clear_keys(self):
        """Clear all stored API keys."""
        try:
            if self.storage_file.exists():
                self.storage_file.unlink()
            logger.info("API keys cleared from storage")
            return True
        except Exception as e:
            logger.error(f"Failed to clear API keys: {e}")
            return False
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key from storage or environment."""
        keys = self.load_keys()
        return keys.get('openai') or os.getenv("OPENAI_API_KEY")
    
    def get_perplexity_key(self) -> Optional[str]:
        """Get Perplexity API key from storage or environment."""
        keys = self.load_keys()
        return keys.get('perplexity') or os.getenv("PERPLEXITY_API_KEY")