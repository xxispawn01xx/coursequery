#!/usr/bin/env python3
"""
Test Windows path detection for course directory.
"""

import os
from pathlib import Path, WindowsPath
from config import Config

def test_windows_paths():
    """Test different Windows path formats."""
    print(" Testing Windows Path Detection")
    print("=" * 40)
    
    # Test different path formats
    paths_to_test = [
        r"H:\Archive Classes",
        "H:/Archive Classes", 
        Path("H:/Archive Classes"),
        Path(r"H:\Archive Classes")
    ]
    
    print("Current working directory:", os.getcwd())
    print("Python platform:", os.name)
    
    for i, path in enumerate(paths_to_test):
        print(f"\nTest {i+1}: {type(path).__name__} - {path}")
        try:
            p = Path(path)
            print(f"  Exists: {p.exists()}")
            print(f"  Is absolute: {p.is_absolute()}")
            print(f"  Resolved: {p.resolve()}")
            
            if p.exists():
                print(f"  Contents preview:")
                try:
                    contents = list(p.iterdir())[:5]  # First 5 items
                    for item in contents:
                        print(f"    - {item.name}")
                except PermissionError:
                    print("    Permission denied")
                except Exception as e:
                    print(f"    Error: {e}")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test config
    print(f"\n Config Test:")
    config = Config()
    print(f"  Raw docs dir: {config.raw_docs_dir}")
    print(f"  Exists: {config.raw_docs_dir.exists()}")
    print(f"  Is Replit: {config.is_replit}")
    print(f"  Skip model loading: {config.skip_model_loading}")

if __name__ == "__main__":
    test_windows_paths()