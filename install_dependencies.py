#!/usr/bin/env python3
"""
Installation script for Real Estate AI Stack dependencies.
Run this script to install all required packages for 100% local operation.
"""

import subprocess
import sys
import importlib

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Main installation process."""
    print("🏘️ Real Estate AI Stack - Installing Dependencies")
    print("=" * 50)
    
    # Core dependencies (required for basic functionality)
    core_packages = [
        "plotly>=5.17.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "beautifulsoup4>=4.12.0",
    ]
    
    # Document processing dependencies
    document_packages = [
        "PyPDF2>=3.0.0",
        "python-docx>=1.1.0", 
        "python-pptx>=0.6.23",
        "ebooklib>=0.18",
    ]
    
    # AI/ML dependencies (for local models)
    ai_packages = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "accelerate>=0.24.0",
        "openai-whisper>=20231117",
        "llama-index>=0.10.0",
        "llama-index-core>=0.10.0",
        "llama-index-embeddings-huggingface>=0.1.0",
    ]
    
    # Optional dependencies (for enhanced functionality)
    optional_packages = [
        "bitsandbytes>=0.41.0",  # GPU quantization
        "ffmpeg-python>=0.2.0",  # Video processing
        "scipy>=1.11.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ]
    
    all_packages = [
        ("Core packages", core_packages),
        ("Document processing", document_packages), 
        ("AI/ML models", ai_packages),
        ("Optional enhancements", optional_packages),
    ]
    
    for category, packages in all_packages:
        print(f"\n📦 Installing {category}...")
        for package in packages:
            install_package(package)
    
    print(f"\n🎉 Installation complete!")
    print("\nTo start the app, run:")
    print("streamlit run app.py --server.port 5000")

if __name__ == "__main__":
    main()