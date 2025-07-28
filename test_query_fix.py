#!/usr/bin/env python3
"""
Quick test to verify the query engine fix.
Run this locally to test if the MockLLM issue is resolved.
"""

import logging
from local_models import LocalModelManager
from query_engine import LocalQueryEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_engine():
    """Test if query engine works without MockLLM errors."""
    print("🔧 Testing Query Engine Fix...")
    
    try:
        # Initialize model manager
        print("📦 Initializing model manager...")
        model_manager = LocalModelManager()
        
        # Load models
        print("🚀 Loading models...")
        model_manager.load_models()
        
        # Initialize query engine
        print("🔍 Initializing query engine...")
        query_engine = LocalQueryEngine(model_manager)
        
        # Test basic response generation
        print("💭 Testing response generation...")
        test_prompt = "What is real estate?"
        response = model_manager.generate_response(test_prompt, max_new_tokens=100)
        
        print(f"✅ Success! Generated response: {response[:100]}...")
        print("🎉 Query engine is working properly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_query_engine()
    exit(0 if success else 1)