#!/usr/bin/env python3
"""
Test business valuation with CUDA error handling.
Quick test to verify the system works for your valuation request.
"""

import os
import sys

# Set CUDA environment for RTX 3060 stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def test_valuation_query():
    """Test a simple valuation query to verify CUDA fix."""
    try:
        from local_models import LocalModelManager
        from config import Config
        
        print("🧪 Testing business valuation with CUDA error handling...")
        
        config = Config()
        model_manager = LocalModelManager()
        
        # Test simple query that should trigger Excel generation
        test_query = "Create a simple business valuation framework with key metrics"
        
        print(f"📝 Test query: {test_query}")
        print("🔄 Loading models (this may take a moment)...")
        
        # Load models
        model_manager.load_models()
        
        print("✅ Models loaded successfully")
        print("🤔 Generating response...")
        
        # Generate response with CUDA error handling
        response = model_manager.generate_response(test_query)
        
        print("✅ Response generated successfully!")
        print(f"📄 Response preview: {response[:200]}...")
        
        # Check if response contains valuation content
        valuation_keywords = ['valuation', 'financial', 'dcf', 'cash flow', 'analysis']
        found_keywords = [kw for kw in valuation_keywords if kw.lower() in response.lower()]
        
        if found_keywords:
            print(f"✅ Valuation content detected: {found_keywords}")
            print("📊 Excel generation feature should be available in the app")
        else:
            print("⚠️ No specific valuation content detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if "CUDA" in str(e):
            print("🔧 CUDA error - the error handling should have prevented this")
        return False

if __name__ == "__main__":
    print("🎯 RTX 3060 Business Valuation Test")
    print("=" * 50)
    
    success = test_valuation_query()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("🚀 Your valuation feature should work in the main app")
        print("📊 Look for the 'Generate Excel File' button after responses")
    else:
        print("\n❌ Test failed - check the error messages above")
        print("🔄 Try restarting the application or using shorter queries")