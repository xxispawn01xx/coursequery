#!/usr/bin/env python3
"""
Test script to verify course detection functionality.
Run this to debug course refresh issues.
"""

import logging
from pathlib import Path
from config import Config
from course_indexer import CourseIndexer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_course_detection():
    """Test the course detection functionality."""
    print("🧪 Testing Course Detection")
    print("=" * 50)
    
    # Initialize components
    config = Config()
    indexer = CourseIndexer()
    
    # Test directory existence
    print(f"📁 Raw docs directory: {config.raw_docs_dir}")
    print(f"   Exists: {config.raw_docs_dir.exists()}")
    
    print(f"📊 Indexed courses directory: {config.indexed_courses_dir}")  
    print(f"   Exists: {config.indexed_courses_dir.exists()}")
    
    # List contents of raw_docs
    if config.raw_docs_dir.exists():
        print("\n📂 Contents of raw_docs:")
        for item in config.raw_docs_dir.iterdir():
            if item.is_dir():
                files = list(item.iterdir())
                print(f"  📁 {item.name}/ ({len(files)} files)")
                for file in files:
                    print(f"    - {file.name}")
            else:
                print(f"  📄 {item.name}")
    
    # List contents of indexed_courses
    if config.indexed_courses_dir.exists():
        print("\n📊 Contents of indexed_courses:")
        for item in config.indexed_courses_dir.iterdir():
            if item.is_dir():
                files = list(item.iterdir())
                print(f"  📁 {item.name}/ ({len(files)} files)")
                for file in files:
                    print(f"    - {file.name}")
    
    # Test course detection
    print("\n🔍 Running course detection...")
    courses = indexer.get_available_courses()
    
    print(f"\n✅ Detection Result: {len(courses)} courses found")
    for course in courses:
        print(f"  - {course['name']} ({course['status']}) - {course['document_count']} docs")
    
    return courses

if __name__ == "__main__":
    courses = test_course_detection()