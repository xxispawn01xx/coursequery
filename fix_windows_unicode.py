#!/usr/bin/env python3
"""
Windows Unicode Compatibility Fix
Removes all Unicode emoji characters that cause cp1252 encoding errors on Windows console
"""
import os
import re
from pathlib import Path

# Common problematic Unicode emojis in the codebase
EMOJI_PATTERNS = [
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
]

def fix_file_unicode(file_path):
    """Remove emoji characters from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # Remove each emoji pattern
        for emoji in EMOJI_PATTERNS:
            if emoji in content:
                # Replace emoji at start of string/line
                content = re.sub(rf'^{re.escape(emoji)}\s*', '', content, flags=re.MULTILINE)
                # Replace emoji in middle of text  
                content = re.sub(rf'\s*{re.escape(emoji)}\s*', ' ', content)
                changes_made = True
        
        # Save if changes were made
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def scan_and_fix_directory(directory_path):
    """Scan directory for Python files and fix Unicode issues."""
    directory = Path(directory_path)
    fixed_files = []
    
    # Find all Python files
    python_files = list(directory.glob('*.py'))
    
    print(f"Scanning {len(python_files)} Python files in {directory}")
    
    for file_path in python_files:
        if fix_file_unicode(file_path):
            fixed_files.append(str(file_path))
    
    return fixed_files

def main():
    """Main execution function."""
    print("Windows Unicode Compatibility Fix")
    print("=" * 40)
    
    # Fix current directory
    current_dir = Path('.')
    fixed_files = scan_and_fix_directory(current_dir)
    
    if fixed_files:
        print(f"\n✓ Fixed {len(fixed_files)} files:")
        for file in fixed_files:
            print(f"  - {file}")
        print(f"\nYour application should now run on Windows without Unicode errors!")
    else:
        print("\n✓ No emoji characters found - your code is already Windows compatible!")

if __name__ == "__main__":
    main()