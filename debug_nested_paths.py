#!/usr/bin/env python3
"""
Debug script to analyze the nested path structure and show what files
are being found vs processed in the deeply nested course directories.
"""

import logging
from pathlib import Path
from collections import Counter
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_nested_course_structure():
    """Analyze the actual course structure with nested paths."""
    
    # Example path structure from user
    example_path = r"H:\Archive Classes\coursequery\archived_courses\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\2 - The basics of Apache Airflow"
    
    print("🔍 NESTED PATH STRUCTURE ANALYSIS")
    print("=" * 60)
    print(f"📁 Example nested path:")
    print(f"   {example_path}")
    print()
    
    # Break down the path structure
    path_parts = Path(example_path).parts
    print("📂 PATH BREAKDOWN:")
    for i, part in enumerate(path_parts):
        indent = "  " * i
        print(f"{indent}└─ {part}")
    print()
    
    print("⚠️ POTENTIAL ISSUES:")
    print("1. Special characters in folder names: [, ], spaces")
    print("2. Very long path names (Windows has 260 char limit)")
    print("3. Duplicate folder names in nested structure")
    print("4. Multiple spaces and special chars")
    print()
    
    # Show what file types should be found
    try:
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        supported = processor.get_supported_formats()
        
        print("✅ SUPPORTED FORMATS (should be captured):")
        video_formats = [f for f in supported if f in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']]
        subtitle_formats = [f for f in supported if f in ['.vtt', '.srt', '.ass']]
        code_formats = [f for f in supported if f in ['.py', '.js', '.json', '.html']]
        
        print(f"📹 Videos: {', '.join(video_formats)}")
        print(f"📝 Subtitles: {', '.join(subtitle_formats)}")
        print(f"💻 Code: {', '.join(code_formats)}")
        print()
        
    except ImportError:
        print("❌ Could not import DocumentProcessor")
    
    print("🔧 DEBUGGING RECOMMENDATIONS:")
    print("1. Check if Windows path length limits are hit")
    print("2. Verify special characters in paths are handled")
    print("3. Test file.exists() on actual nested paths")
    print("4. Check if rglob() works with complex folder names")
    print()
    
    print("💡 EXPECTED COURSE CONTENT:")
    print("For Apache Airflow Udemy course, you should see:")
    print("• Hundreds of MP4 video files")
    print("• Corresponding VTT subtitle files")
    print("• Python code examples (.py files)")
    print("• Configuration files (.json, .yaml)")
    print("• HTML/CSS examples")
    print("• Course resources (PDFs, docs)")
    print()
    
    print("🎯 LOCAL TESTING SCRIPT:")
    print("Run this on your H:\\ drive to debug:")
    print("""
from pathlib import Path
from collections import Counter

# Point to your actual course
base_path = Path(r'H:\\Archive Classes\\coursequery\\archived_courses')
course_name = '[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide'
course_path = base_path / course_name

print(f'Checking: {course_path}')
print(f'Exists: {course_path.exists()}')

if course_path.exists():
    # Count all files recursively
    all_files = list(course_path.rglob('*'))
    files_only = [f for f in all_files if f.is_file()]
    
    print(f'Total files found: {len(files_only)}')
    
    # Show file type breakdown
    extensions = Counter(f.suffix.lower() for f in files_only)
    print('File types:')
    for ext, count in extensions.most_common():
        print(f'  {ext}: {count} files')
    
    # Test specific file paths
    print('\\nTesting specific nested paths:')
    for section in course_path.iterdir():
        if section.is_dir():
            print(f'  Section: {section.name}')
            section_files = list(section.rglob('*.mp4'))
            print(f'    MP4 files: {len(section_files)}')
            if section_files:
                test_file = section_files[0]
                print(f'    Test file exists: {test_file.exists()}')
                print(f'    Test file path: {test_file}')
""")

if __name__ == "__main__":
    analyze_nested_course_structure()