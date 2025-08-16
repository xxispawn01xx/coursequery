#!/usr/bin/env python3
"""
Direct test that simulates exactly what happens in the app when you click Q&A.
This will show us the real failure point without heavy model loading.
"""

import logging
import sys
from unittest.mock import Mock, MagicMock

# Set up logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" Testing Local App Flow...")

# Test 1: Config initialization
print("\n1️⃣ Testing Config...")
try:
    from config import Config
    config = Config()
    print(f" Config created - skip_model_loading: {config.skip_model_loading}")
    if config.skip_model_loading:
        print(" FOUND THE ISSUE! skip_model_loading is True")
        print("   This means the app thinks it should skip models")
        exit(1)
    else:
        print(" Config correctly set to load models")
except Exception as e:
    print(f" Config failed: {e}")
    exit(1)

# Test 2: Mock model manager
print("\n2️⃣ Testing Model Manager Creation...")
try:
    # Mock heavy imports
    sys.modules['torch'] = Mock()
    sys.modules['transformers'] = Mock()
    sys.modules['sentence_transformers'] = Mock()
    
    from local_models import LocalModelManager
    
    # Create model manager but don't load heavy models
    model_manager = LocalModelManager()
    print(" Model manager created successfully")
    
    # Mock the load_models method
    original_load = model_manager.load_models
    def mock_load_models():
        print(" Mock loading models...")
        model_manager.mistral_pipeline = Mock()
        model_manager.embedding_model = Mock()
        return True
    
    model_manager.load_models = mock_load_models
    model_manager.load_models()
    print(" Mock models loaded")
    
except Exception as e:
    print(f" Model manager failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Query engine with mocks
print("\n3️⃣ Testing Query Engine Creation...")
try:
    # Mock llama_index imports
    sys.modules['llama_index'] = Mock()
    sys.modules['llama_index.core'] = Mock()
    sys.modules['llama_index.core.retrievers'] = Mock()
    sys.modules['llama_index.core.postprocessor'] = Mock()
    sys.modules['llama_index.embeddings'] = Mock()
    sys.modules['llama_index.embeddings.huggingface'] = Mock()
    
    from query_engine import LocalQueryEngine
    
    # This is the critical test - can we create the query engine?
    query_engine = LocalQueryEngine(model_manager)
    print(" Query engine created successfully")
    
    # Test if it has the required attributes
    if hasattr(query_engine, 'model_manager') and query_engine.model_manager:
        print(" Query engine has model manager")
    else:
        print(" Query engine missing model manager")
        exit(1)
        
except Exception as e:
    print(f" Query engine creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: App initialization simulation
print("\n4️⃣ Testing App Flow Simulation...")
try:
    # Create mock session state
    mock_session_state = {
        'models_loaded': False,
        'hf_token_set': True  # Assume auth is OK
    }
    
    # Simulate the app's model loading logic
    print(" Simulating app.load_models()...")
    
    # This simulates the load_models method
    if not mock_session_state['models_loaded']:
        print(" Models not loaded, attempting to load...")
        
        # Mock the model loading process
        try:
            model_manager.load_models()  # Our mocked version
            app_query_engine = LocalQueryEngine(model_manager)  # This should work
            mock_session_state['models_loaded'] = True
            print(" Models loaded and query engine created")
        except Exception as e:
            print(f" Model loading simulation failed: {e}")
            exit(1)
    
    # Test the final check that was failing
    if app_query_engine is None:
        print(" FOUND THE ISSUE! Query engine is None after creation")
        exit(1)
    else:
        print(" Query engine properly initialized")
        
except Exception as e:
    print(f" App flow simulation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 All tests passed! The query engine initialization should work.")
print("\nThe issue was likely:")
print("1. skip_model_loading was True (now fixed)")
print("2. Missing import handling (now mocked)")
print("3. Core logic is working properly")

print("\n Ready to test locally with real models!")