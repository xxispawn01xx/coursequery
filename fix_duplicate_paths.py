#!/usr/bin/env python3
"""
Fix duplicate nested directory paths in course structure
This addresses the duplicate course name issue in file paths
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def detect_duplicate_paths(file_path):
    """
    Detect and fix duplicate nested directory structures.
    
    Example problematic path:
    archived_courses\[Course Name]\[Course Name]\...
    
    Should be:
    archived_courses\[Course Name]\...
    """
    
    path = Path(file_path)
    parts = path.parts
    
    print(f"Original path: {file_path}")
    print(f"Path parts: {parts}")
    
    # Look for duplicate consecutive directories
    duplicates_found = []
    fixed_parts = list(parts)
    
    for i in range(len(parts) - 1):
        if parts[i] == parts[i + 1]:
            duplicates_found.append((i, parts[i]))
            print(f"Found duplicate at position {i}: {parts[i]}")
    
    # Remove duplicates (working backwards to preserve indices)
    for i, duplicate_name in reversed(duplicates_found):
        print(f"Removing duplicate: {duplicate_name}")
        fixed_parts.pop(i + 1)  # Remove the second occurrence
    
    if duplicates_found:
        fixed_path = Path(*fixed_parts)
        print(f"Fixed path: {fixed_path}")
        print(f"Fixed exists: {fixed_path.exists()}")
        return str(fixed_path)
    else:
        print("No duplicates found")
        return file_path

def search_for_actual_file(file_name, search_root="archived_courses"):
    """Search for the actual location of a file."""
    print(f"\nSearching for: {file_name}")
    print(f"Search root: {search_root}")
    
    search_path = Path(search_root)
    if not search_path.exists():
        print(f"Search root doesn't exist: {search_path}")
        return None
    
    # Search for the file
    found_files = list(search_path.rglob(file_name))
    
    print(f"Found {len(found_files)} matches:")
    for found_file in found_files:
        print(f"  - {found_file}")
        print(f"    Exists: {found_file.exists()}")
        print(f"    Is file: {found_file.is_file()}")
        if found_file.exists() and found_file.is_file():
            return str(found_file)
    
    return None

if __name__ == "__main__":
    # Test with the problematic path
    test_path = r"archived_courses\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\2 - The basics of Apache Airflow\12 - Practice Quick tour of Airflow CLI.mp4"
    
    print("Duplicate Path Fix Test")
    print("=" * 50)
    
    # Test duplicate detection
    fixed_path = detect_duplicate_paths(test_path)
    
    # Test file search
    file_name = "12 - Practice Quick tour of Airflow CLI.mp4"
    actual_location = search_for_actual_file(file_name)
    
    if actual_location:
        print(f"\n✅ FOUND: {actual_location}")
    else:
        print(f"\n❌ File not found: {file_name}")
        
        # Also try searching for any MP4 files
        print("\nSearching for any MP4 files...")
        search_path = Path("archived_courses")
        if search_path.exists():
            mp4_files = list(search_path.rglob("*.mp4"))
            print(f"Found {len(mp4_files)} MP4 files in total:")
            for mp4 in mp4_files[:5]:  # Show first 5
                print(f"  - {mp4}")
            if len(mp4_files) > 5:
                print(f"  ... and {len(mp4_files) - 5} more")