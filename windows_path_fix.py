#!/usr/bin/env python3
"""
Windows path fix for transcription issues
This addresses the specific WinError 2 "system cannot find file specified" issue
"""

import os
import sys
from pathlib import Path

def fix_windows_path(file_path):
    """
    Fix Windows path issues for Whisper transcription.
    
    The issue is that Windows paths with backslashes and special characters
    in course names aren't being handled properly by Whisper.
    """
    
    # Convert to Path object for proper handling
    path = Path(file_path)
    
    # Method 1: Use forward slashes (Unix-style paths work on Windows)
    forward_slash_path = str(path).replace('\\', '/')
    
    # Method 2: Use raw string path 
    raw_path = str(path.resolve())
    
    # Method 3: Use os.path.normpath
    normalized_path = os.path.normpath(str(path))
    
    # Method 4: Use short path name if available (Windows 8.3 format)
    try:
        import win32api
        short_path = win32api.GetShortPathName(str(path))
    except:
        short_path = None
    
    return {
        'original': str(path),
        'forward_slash': forward_slash_path,
        'raw': raw_path,
        'normalized': normalized_path,
        'short_path': short_path,
        'exists': path.exists(),
        'resolved': str(path.resolve()) if path.exists() else None
    }

def test_file_access(file_path):
    """Test different ways to access a file."""
    print(f"Testing file access for: {file_path}")
    
    results = fix_windows_path(file_path)
    
    for method, path_str in results.items():
        if path_str and method != 'exists':
            try:
                if method == 'exists':
                    continue
                    
                # Test if file is accessible
                with open(path_str, 'rb') as f:
                    # Just read first few bytes to test access
                    data = f.read(10)
                    print(f"  ✅ {method}: ACCESSIBLE - {path_str}")
                    return path_str, method
                    
            except Exception as e:
                print(f"  ❌ {method}: {e}")
    
    return None, None

if __name__ == "__main__":
    # Test with the problematic path from your logs
    test_path = r"archived_courses\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\2 - The basics of Apache Airflow\12 - Practice Quick tour of Airflow CLI.mp4"
    
    print("Windows Path Fix Test")
    print("=" * 50)
    
    results = fix_windows_path(test_path)
    
    print("Path conversion results:")
    for method, result in results.items():
        print(f"  {method}: {result}")
    
    print("\nAccess test:")
    working_path, working_method = test_file_access(test_path)
    
    if working_path:
        print(f"\n✅ SOLUTION: Use {working_method} format")
        print(f"Working path: {working_path}")
    else:
        print(f"\n❌ No method worked - file may not exist or be inaccessible")