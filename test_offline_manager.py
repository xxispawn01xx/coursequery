#!/usr/bin/env python3
"""
Test the offline course manager
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_offline_manager():
    """Test offline course manager functionality"""
    logger.info("🧪 Testing offline course manager...")
    
    try:
        from offline_course_manager import OfflineCourseManager
        
        # Initialize with app directories
        app_dir = Path(__file__).parent
        raw_docs = app_dir / "archived_courses"
        indexed_courses = app_dir / "indexed_courses"
        
        manager = OfflineCourseManager(raw_docs, indexed_courses)
        
        # Test course detection
        logger.info(" Testing course detection...")
        courses = manager.get_available_courses()
        
        logger.info(f" Found {len(courses)} courses:")
        for course in courses:
            logger.info(f"  - {course['name']} ({course['status']}) - {course.get('document_count', 0)} docs")
        
        logger.info(" Offline course manager test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f" Offline course manager test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_offline_manager()