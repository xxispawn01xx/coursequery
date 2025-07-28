#!/usr/bin/env python3
"""
Diagnostic tool to identify what's causing model loading delays.
Run this locally to see exactly where time is being spent.
"""

import time
from pathlib import Path
import os

def check_cache_status():
    """Check what models are actually cached."""
    print("🔍 Checking model cache status...")
    
    models_dir = Path("./models")
    if not models_dir.exists():
        print("❌ Models directory doesn't exist")
        return
    
    print(f"📁 Models directory: {models_dir.absolute()}")
    print(f"   Total size: {get_dir_size(models_dir) / (1024**3):.2f} GB")
    
    # Check for common model cache patterns
    cache_patterns = [
        "models--meta-llama--Llama-2-7b-chat-hf",
        "models--mistralai--Mistral-7B-Instruct-v0.1", 
        "models--sentence-transformers--all-MiniLM-L6-v2"
    ]
    
    for pattern in cache_patterns:
        cache_path = models_dir / pattern
        if cache_path.exists():
            size_gb = get_dir_size(cache_path) / (1024**3)
            print(f"✅ {pattern}: {size_gb:.2f} GB")
            
            # Check for snapshots (actual model files)
            snapshots = cache_path / "snapshots"
            if snapshots.exists():
                snapshot_dirs = list(snapshots.iterdir())
                if snapshot_dirs:
                    print(f"   📄 Snapshots: {len(snapshot_dirs)} versions")
                    latest = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                    print(f"   🕒 Latest: {latest.name}")
                else:
                    print("   ❌ No snapshots found (incomplete download)")
            else:
                print("   ❌ No snapshots directory")
        else:
            print(f"❌ {pattern}: Not cached")
    
    # Check .locks directory
    locks_dir = models_dir / ".locks"
    if locks_dir.exists():
        lock_files = list(locks_dir.glob("*"))
        if lock_files:
            print(f"⚠️  Lock files present: {len(lock_files)} (indicates downloads in progress)")
            for lock in lock_files[:3]:  # Show first 3
                print(f"   🔒 {lock.name}")
        else:
            print("✅ No lock files (no downloads in progress)")

def get_dir_size(path):
    """Get total size of directory."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except (OSError, FileNotFoundError):
        pass
    return total

def time_model_loading_steps():
    """Time each step of model loading to identify bottlenecks."""
    print("\n⏱️  Timing model loading steps...")
    
    try:
        # Step 1: Import timing
        start = time.time()
        from local_models import LocalModelManager
        import_time = time.time() - start
        print(f"📦 Import time: {import_time:.2f}s")
        
        # Step 2: Initialization timing
        start = time.time()
        manager = LocalModelManager()
        init_time = time.time() - start
        print(f"🔧 Init time: {init_time:.2f}s")
        
        # Step 3: Check if models already loaded
        start = time.time()
        already_loaded = manager._models_already_loaded("mistral")
        check_time = time.time() - start
        print(f"🔍 Cache check time: {check_time:.2f}s")
        print(f"   Already loaded: {already_loaded}")
        
        if not already_loaded:
            print("\n⚠️  Models not in memory - this will trigger full loading")
            print("💡 For faster startup, keep the app running or implement model persistence")
        else:
            print("\n✅ Models already in memory - should be fast!")
            
    except Exception as e:
        print(f"❌ Error during timing: {e}")

def check_environment():
    """Check environment factors that affect loading speed."""
    print("\n🖥️  Environment check...")
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"✅ HF Token: {hf_token[:7]}...")
    else:
        print("❌ No HF Token found")
        
    # Check if .hf_token file exists
    hf_file = Path(".hf_token")
    if hf_file.exists():
        print("✅ .hf_token file exists")
    else:
        print("❌ .hf_token file missing")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"🚀 CUDA: {device_count} device(s) - {device_name}")
        else:
            print("⚠️  CUDA: Not available (CPU mode will be slower)")
    except ImportError:
        print("❌ PyTorch not available")

def main():
    print("🔧 Model Loading Diagnostic Tool")
    print("=" * 50)
    
    check_environment()
    check_cache_status()
    time_model_loading_steps()
    
    print("\n💡 Optimization Tips:")
    print("1. Keep the app running to avoid reloading models")
    print("2. Ensure models are fully cached (no lock files)")
    print("3. Use GPU if available for faster loading")
    print("4. Check if any antivirus is scanning the models directory")
    print("5. Consider using smaller models for development")

if __name__ == "__main__":
    main()