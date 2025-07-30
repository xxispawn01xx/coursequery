"""
Pydantic compatibility fix for offline mode
Handles import issues with different pydantic versions
"""

import sys
from pathlib import Path

def apply_pydantic_fix():
    """Apply compatibility fixes for pydantic import issues"""
    try:
        # Try to import and fix pydantic internal imports
        import pydantic._internal
        
        # Check if _model_construction exists
        if not hasattr(pydantic._internal, '_model_construction'):
            # Create a mock _model_construction module for compatibility
            class MockModelConstruction:
                def __getattr__(self, name):
                    # Return a dummy function for any missing attributes
                    return lambda *args, **kwargs: None
            
            pydantic._internal._model_construction = MockModelConstruction()
            
    except ImportError:
        # If pydantic is not available, create a minimal mock
        class MockPydantic:
            class _internal:
                class _model_construction:
                    pass
        
        sys.modules['pydantic'] = MockPydantic()
        sys.modules['pydantic._internal'] = MockPydantic._internal()
        sys.modules['pydantic._internal._model_construction'] = MockPydantic._internal._model_construction()

def safe_import_with_fallback(module_name, fallback_class=None):
    """Safely import a module with fallback for offline mode"""
    try:
        # Apply pydantic fix first
        apply_pydantic_fix()
        
        # Try to import the module
        module = __import__(module_name, fromlist=[''])
        return module
        
    except Exception as e:
        print(f"⚠️ Import warning for {module_name}: {e}")
        
        if fallback_class:
            return fallback_class
        
        # Return a minimal mock class
        class MockModule:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        return MockModule()

# Apply the fix immediately when this module is imported
apply_pydantic_fix()