#!/usr/bin/env python3
"""
Debug script to isolate course loading issues for offline app
"""

import logging
import time
from pathlib import Path
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def debug_course_loading():
    """Debug course loading step by step"""
    logger.info(" Starting debug course loading for offline app...")
    
    try:
        # Step 1: Check basic imports
        logger.info(" Step 1: Testing basic imports...")
        
        try:
            from config import Config
            logger.info(" Config import successful")
        except Exception as e:
            logger.error(f" Config import failed: {e}")
            return
        
        try:
            from course_indexer import CourseIndexer
            logger.info(" CourseIndexer import successful")
        except Exception as e:
            logger.error(f" CourseIndexer import failed: {e}")
            return
        
        # Step 2: Initialize configuration
        logger.info(" Step 2: Initializing configuration...")
        config = Config()
        
        # Force local directories
        app_dir = Path(__file__).parent
        config.raw_docs_dir = app_dir / "archived_courses"
        config.indexed_courses_dir = app_dir / "indexed_courses"
        
        logger.info(f" Raw docs directory: {config.raw_docs_dir}")
        logger.info(f" Indexed courses directory: {config.indexed_courses_dir}")
        logger.info(f" Raw docs exists: {config.raw_docs_dir.exists()}")
        logger.info(f" Indexed courses exists: {config.indexed_courses_dir.exists()}")
        
        # Step 3: Initialize course indexer
        logger.info(" Step 3: Initializing course indexer...")
        start_time = time.time()
        
        course_indexer = CourseIndexer()
        init_time = time.time() - start_time
        logger.info(f" CourseIndexer initialized in {init_time:.2f} seconds")
        
        # Step 4: Get available courses (this is where it might hang)
        logger.info(" Step 4: Getting available courses...")
        start_time = time.time()
        
        # Add cross-platform timeout for this operation
        import threading
        
        try:
            courses = course_indexer.get_available_courses()
            courses_time = time.time() - start_time
            logger.info(f" Found {len(courses)} courses in {courses_time:.2f} seconds")
            
            # Log course details
            for i, course in enumerate(courses[:5]):  # Show first 5 courses
                logger.info(f"  {i+1}. Course: {course.get('name', 'Unknown')}")
                logger.info(f"     Status: {course.get('status', 'Unknown')}")
                logger.info(f"     Document count: {course.get('document_count', 'Unknown')}")
                
        except TimeoutError:
            logger.error("⏰ TIMEOUT: get_available_courses took longer than 60 seconds")
            return
        finally:
            signal.alarm(0)
        
        # Step 5: Test loading a specific course index (if any exist)
        indexed_courses = [c for c in courses if c.get('status') == 'indexed']
        if indexed_courses:
            test_course = indexed_courses[0]
            course_name = test_course['name']
            
            logger.info(f"🧪 Step 5: Testing index load for course: {course_name}")
            start_time = time.time()
            
            signal.alarm(60)  # 60 second timeout
            try:
                index = course_indexer.get_course_index(course_name)
                load_time = time.time() - start_time
                
                if index:
                    logger.info(f" Successfully loaded index in {load_time:.2f} seconds")
                else:
                    logger.warning(f" Index returned None in {load_time:.2f} seconds")
                    
            except TimeoutError:
                logger.error(f"⏰ TIMEOUT: Loading index for {course_name} took longer than 60 seconds")
            except Exception as e:
                logger.error(f" Error loading index for {course_name}: {e}")
            finally:
                signal.alarm(0)
        else:
            logger.info("ℹ️ No indexed courses found to test")
        
    except Exception as e:
        logger.error(f" Debug failed with error: {e}")
        import traceback
        logger.error(f" Full traceback:\n{traceback.format_exc()}")
    
    logger.info("🏁 Debug course loading completed")

if __name__ == "__main__":
    debug_course_loading()