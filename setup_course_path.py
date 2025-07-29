#!/usr/bin/env python3
"""
Setup script to configure the course directory path.
Run this to connect to your actual course folders.
"""

import os
from pathlib import Path
from config import Config

def setup_course_directory():
    """Configure the course directory path."""
    print("🔧 Course Directory Setup")
    print("=" * 40)
    
    # User's actual course directory
    user_path = "H:/Archive Classes"
    
    print(f"Checking for course directory: {user_path}")
    
    if Path(user_path).exists():
        print(f"✅ Found your course directory!")
        
        # List courses found
        course_path = Path(user_path)
        courses = []
        for item in course_path.iterdir():
            if item.is_dir():
                # Count supported files in each course
                supported_files = []
                for file_path in item.rglob('*'):
                    if file_path.is_file():
                        supported_extensions = ['.pdf', '.docx', '.pptx', '.epub', '.mp4', '.avi', '.mov', '.mp3', '.wav']
                        if file_path.suffix.lower() in supported_extensions:
                            supported_files.append(file_path)
                
                if supported_files:
                    courses.append({
                        'name': item.name,
                        'path': str(item),
                        'files': len(supported_files)
                    })
        
        print(f"\n📚 Found {len(courses)} courses with supported files:")
        for course in courses[:10]:  # Show first 10
            print(f"  - {course['name']} ({course['files']} files)")
        
        if len(courses) > 10:
            print(f"  ... and {len(courses) - 10} more courses")
        
        # Test configuration
        config = Config()
        print(f"\n🎯 Configuration test:")
        print(f"  Raw docs directory: {config.raw_docs_dir}")
        print(f"  Indexed courses directory: {config.indexed_courses_dir}")
        
        return True
    else:
        print(f"❌ Course directory not found: {user_path}")
        print("\nCurrent working directory contents:")
        for item in Path(".").iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
        
        print("\nTo use your actual courses:")
        print("1. Update the path in config.py")
        print("2. Or create a symbolic link to your course directory") 
        return False

if __name__ == "__main__":
    setup_course_directory()