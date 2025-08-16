#!/usr/bin/env python3
"""
Quick memory test to check if system can handle AI models.
Run this before trying to load large models.
"""

import psutil
import sys

def check_system_memory():
    """Check if system has enough memory for AI models."""
    print(" Checking system memory...")
    
    # Get memory info
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)
    swap_total_gb = swap.total / (1024**3)
    
    print(f" Total RAM: {total_gb:.1f} GB")
    print(f" Available RAM: {available_gb:.1f} GB")
    print(f" Used RAM: {used_gb:.1f} GB ({memory.percent:.1f}%)")
    print(f" Virtual Memory: {swap_total_gb:.1f} GB")
    
    # Recommendations
    print(f"\n🤖 Model Recommendations:")
    
    if available_gb >= 8.0:
        print(" Can handle Mistral 7B (7GB+)")
        recommended = "mistral"
    elif available_gb >= 2.0:
        print(" Can handle DialoGPT Medium (330MB)")
        print(" Mistral 7B may cause memory issues")
        recommended = "medium"
    elif available_gb >= 1.0:
        print(" Can handle DialoGPT Small (120MB)")
        print(" Larger models will likely fail")
        recommended = "small"
    else:
        print(" Very low memory - consider closing other programs")
        recommended = "small"
    
    # Virtual memory check
    if swap_total_gb < 4.0:
        print(f"\n Virtual memory is low ({swap_total_gb:.1f}GB)")
        print(" Consider increasing virtual memory (paging file) on Windows")
        print("   System Properties → Advanced → Virtual Memory → Custom Size")
    
    return recommended

def main():
    print("🧠 AI Model Memory Compatibility Test")
    print("=" * 40)
    
    try:
        recommended = check_system_memory()
        
        print(f"\n Recommended setting: {recommended}")
        print("\n Next steps:")
        print("1. Use the recommended model size")
        print("2. Close unnecessary programs to free memory")
        print("3. Increase virtual memory if needed")
        print("4. Consider restarting your computer to clear memory")
        
    except Exception as e:
        print(f" Error checking memory: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())