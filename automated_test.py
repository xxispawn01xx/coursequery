#!/usr/bin/env python3
"""
Automated test to diagnose query engine initialization without loading heavy models.
This simulates the app flow to find exactly where initialization fails.
"""

import logging
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock Streamlit to avoid import issues
sys.modules['streamlit'] = Mock()
sys.modules['plotly'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()
sys.modules['pandas'] = Mock()

def test_imports():
    """Test if all required modules can be imported."""
    print(" Testing imports...")
    
    try:
        from config import Config
        print(" Config imported successfully")
        
        from local_models import LocalModelManager
        print(" LocalModelManager imported successfully")
        
        from query_engine import LocalQueryEngine
        print(" LocalQueryEngine imported successfully")
        
        from course_indexer import CourseIndexer
        print(" CourseIndexer imported successfully")
        
        return True
    except Exception as e:
        print(f" Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_initialization():
    """Test config initialization."""
    print("\n Testing config initialization...")
    
    try:
        from config import Config
        config = Config()
        
        print(f" Config initialized")
        print(f"   - Base dir: {config.base_dir}")
        print(f"   - Is Replit: {config.is_replit}")
        print(f"   - Skip model loading: {config.skip_model_loading}")
        
        return config
    except Exception as e:
        print(f" Config initialization failed: {e}")
        traceback.print_exc()
        return None

def test_mock_model_manager():
    """Test model manager initialization with mocked heavy operations."""
    print("\n🤖 Testing model manager with mocks...")
    
    try:
        # Mock heavy operations
        with patch('local_models.TRANSFORMERS_AVAILABLE', True), \
             patch('local_models.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('local_models.torch.cuda.is_available', return_value=False):
            
            from local_models import LocalModelManager
            
            # Mock the model loading methods
            original_load_mistral = LocalModelManager._load_mistral_model
            original_load_embedding = LocalModelManager._load_embedding_model
            
            def mock_load_mistral(self):
                print("   🔹 Mock loading Mistral model...")
                self.mistral_model = Mock()
                self.mistral_tokenizer = Mock()
                self.mistral_pipeline = Mock()
                self.mistral_pipeline.return_value = [{"generated_text": "Mock response"}]
                
            def mock_load_embedding(self):
                print("   🔹 Mock loading embedding model...")
                self.embedding_model = Mock()
                self.embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
            
            LocalModelManager._load_mistral_model = mock_load_mistral
            LocalModelManager._load_embedding_model = mock_load_embedding
            
            # Test initialization
            model_manager = LocalModelManager()
            print(" Model manager initialized")
            
            # Test model loading
            model_manager.load_models()
            print(" Mock models loaded")
            
            # Test response generation
            response = model_manager.generate_response("Test prompt", max_new_tokens=50)
            print(f" Mock response generated: {response}")
            
            # Restore original methods
            LocalModelManager._load_mistral_model = original_load_mistral
            LocalModelManager._load_embedding_model = original_load_embedding
            
            return model_manager
            
    except Exception as e:
        print(f" Model manager test failed: {e}")
        traceback.print_exc()
        return None

def test_course_indexer():
    """Test course indexer initialization."""
    print("\n Testing course indexer...")
    
    try:
        from course_indexer import CourseIndexer
        
        indexer = CourseIndexer()
        print(" Course indexer initialized")
        
        # Test with mock model manager
        mock_model_manager = Mock()
        mock_model_manager.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        indexer.set_model_manager(mock_model_manager)
        print(" Model manager set on indexer")
        
        return indexer
        
    except Exception as e:
        print(f" Course indexer test failed: {e}")
        traceback.print_exc()
        return None

def test_query_engine():
    """Test query engine initialization with mocks."""
    print("\n Testing query engine...")
    
    try:
        from query_engine import LocalQueryEngine
        
        # Create mock model manager
        mock_model_manager = Mock()
        mock_model_manager.generate_response.return_value = "Mock query response"
        mock_model_manager.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        # Test initialization
        query_engine = LocalQueryEngine(mock_model_manager)
        print(" Query engine initialized")
        
        # Test if it's properly set up
        if hasattr(query_engine, 'model_manager') and query_engine.model_manager:
            print(" Model manager properly set")
        else:
            print(" Model manager not set properly")
            return None
            
        return query_engine
        
    except Exception as e:
        print(f" Query engine test failed: {e}")
        traceback.print_exc()
        return None

def test_app_flow_simulation():
    """Simulate the app flow to find where initialization fails."""
    print("\n🎮 Simulating app flow...")
    
    try:
        # Mock streamlit session state
        mock_session_state = {
            'model_manager': None,
            'query_engine': None,
            'current_course': None,
            'course_index': None
        }
        
        print("   🔹 Step 1: Initialize model manager...")
        mock_model_manager = Mock()
        mock_model_manager.generate_response.return_value = "Test response"
        mock_session_state['model_manager'] = mock_model_manager
        print(" Model manager created")
        
        print("   🔹 Step 2: Initialize query engine...")
        from query_engine import LocalQueryEngine
        query_engine = LocalQueryEngine(mock_model_manager)
        mock_session_state['query_engine'] = query_engine
        print(" Query engine created")
        
        print("   🔹 Step 3: Check if query engine is ready...")
        if mock_session_state['query_engine'] and hasattr(mock_session_state['query_engine'], 'model_manager'):
            print(" Query engine is properly initialized")
            return True
        else:
            print(" Query engine missing or not properly set up")
            return False
            
    except Exception as e:
        print(f" App flow simulation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print(" Starting automated query engine diagnostics...\n")
    
    # Test 1: Imports
    if not test_imports():
        print("\n Import test failed - stopping here")
        return False
    
    # Test 2: Config
    config = test_config_initialization()
    if not config:
        print("\n Config test failed - stopping here") 
        return False
    
    # Test 3: Model Manager
    model_manager = test_mock_model_manager()
    if not model_manager:
        print("\n Model manager test failed - stopping here")
        return False
    
    # Test 4: Course Indexer
    indexer = test_course_indexer()
    if not indexer:
        print("\n Course indexer test failed - stopping here")
        return False
    
    # Test 5: Query Engine
    query_engine = test_query_engine()
    if not query_engine:
        print("\n Query engine test failed - stopping here")
        return False
    
    # Test 6: App Flow
    if not test_app_flow_simulation():
        print("\n App flow simulation failed - stopping here")
        return False
    
    print("\n🎉 All tests passed! The issue might be in the Streamlit app logic.")
    print("\nNext steps:")
    print("1. Check app.py for session state management")
    print("2. Look for where the 'Query engine not initialized' error comes from")
    print("3. Verify proper initialization order in the Streamlit app")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)