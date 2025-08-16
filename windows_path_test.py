#!/usr/bin/env python3
"""
Windows path testing script - run this locally on your H:\ drive
to see exactly what files are found vs processed.
"""

from pathlib import Path
from collections import Counter
import os

def test_windows_paths():
    """Test the actual Windows paths and file accessibility."""
    
    print(" WINDOWS PATH TESTING")
    print("=" * 50)
    
    # Your actual course path
    base_path = Path(r'H:\Archive Classes\coursequery\archived_courses')
    course_name = '[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide'
    course_path = base_path / course_name
    
    print(f" Base path: {base_path}")
    print(f" Course name: {course_name}")
    print(f" Full course path: {course_path}")
    print(f"📏 Path length: {len(str(course_path))} characters")
    print()
    
    # Test basic accessibility
    print(" BASIC PATH TESTS:")
    print(f" Base exists: {base_path.exists()}")
    print(f" Course exists: {course_path.exists()}")
    print()
    
    if not course_path.exists():
        print(" Course path not found. Available courses:")
        try:
            for item in base_path.iterdir():
                if item.is_dir():
                    print(f" {item.name}")
        except Exception as e:
            print(f" Cannot list courses: {e}")
        return
    
    # Count all files recursively
    print(" FILE ANALYSIS:")
    try:
        all_files = list(course_path.rglob('*'))
        files_only = [f for f in all_files if f.is_file()]
        
        print(f" Total items found: {len(all_files)}")
        print(f" Files only: {len(files_only)}")
        print()
        
        # File type breakdown
        extensions = Counter(f.suffix.lower() for f in files_only)
        print(" FILE TYPES FOUND:")
        for ext, count in extensions.most_common():
            print(f"  {ext or '(no extension)'}: {count} files")
        print()
        
        # Test specific file types you expect
        video_files = [f for f in files_only if f.suffix.lower() in ['.mp4', '.avi', '.mov']]
        subtitle_files = [f for f in files_only if f.suffix.lower() in ['.vtt', '.srt']]
        code_files = [f for f in files_only if f.suffix.lower() in ['.py', '.js', '.json']]
        
        print(" EXPECTED COURSE CONTENT:")
        print(f"📹 Video files (.mp4, .avi, .mov): {len(video_files)}")
        print(f" Subtitle files (.vtt, .srt): {len(subtitle_files)}")
        print(f" Code files (.py, .js, .json): {len(code_files)}")
        print()
        
        # Test path length issues
        long_paths = [f for f in files_only if len(str(f)) > 250]
        if long_paths:
            print(f" WARNING: {len(long_paths)} files have very long paths (>250 chars)")
            print("These may cause processing issues on Windows.")
            print()
        
        # Test specific nested structure
        print(" SECTION ANALYSIS:")
        sections = [d for d in course_path.iterdir() if d.is_dir()]
        print(f" Found {len(sections)} course sections")
        
        for i, section in enumerate(sections[:5]):  # Show first 5 sections
            section_files = list(section.rglob('*'))
            section_files_only = [f for f in section_files if f.is_file()]
            section_videos = [f for f in section_files_only if f.suffix.lower() == '.mp4']
            
            print(f" {section.name}")
            print(f" Files: {len(section_files_only)}")
            print(f"     📹 Videos: {len(section_videos)}")
            
            # Test if we can access a video file
            if section_videos:
                test_video = section_videos[0]
                print(f" Sample video: {test_video.name}")
                print(f" Accessible: {test_video.exists()}")
                print(f"     📏 Path length: {len(str(test_video))}")
        
        if len(sections) > 5:
            print(f"  ... and {len(sections) - 5} more sections")
        
    except Exception as e:
        print(f" Error analyzing course: {e}")
        return
    
    print()
    print(" DIAGNOSIS:")
    total_expected = len(video_files) + len(subtitle_files) + len(code_files)
    print(f"Expected processable files: {total_expected}")
    print("If you're only seeing 6 documents, the issue is likely:")
    print("1. Path length limits causing file access failures")
    print("2. Special characters in paths causing encoding issues") 
    print("3. File processing errors due to complex nested structure")
    print()
    print(" RECOMMENDATION:")
    print("The expanded file format support should capture all these files.")
    print("Try re-indexing the course to see the improvement!")

if __name__ == "__main__":
    test_windows_paths()