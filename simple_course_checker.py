#!/usr/bin/env python3
"""
Simple course checker for offline app - no complex imports
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_course_directories():
    """Simple check of course directories without complex imports"""
    logger.info("🔍 Checking course directories for offline app...")
    
    # Check basic directories
    app_dir = Path(__file__).parent
    raw_docs = app_dir / "archived_courses"
    indexed_courses = app_dir / "indexed_courses"
    
    logger.info(f"📁 App directory: {app_dir}")
    logger.info(f"📁 Raw docs: {raw_docs} (exists: {raw_docs.exists()})")
    logger.info(f"📁 Indexed: {indexed_courses} (exists: {indexed_courses.exists()})")
    
    # Check contents
    if raw_docs.exists():
        try:
            raw_contents = list(raw_docs.iterdir())
            logger.info(f"📚 Raw docs contains {len(raw_contents)} items:")
            for item in raw_contents[:5]:
                logger.info(f"  - {item.name}")
        except Exception as e:
            logger.error(f"❌ Cannot read raw docs: {e}")
    
    if indexed_courses.exists():
        try:
            indexed_contents = list(indexed_courses.iterdir())
            logger.info(f"📊 Indexed courses contains {len(indexed_contents)} items:")
            for item in indexed_contents[:5]:
                logger.info(f"  - {item.name}")
                
                # Check if it's a valid course directory
                if item.is_dir():
                    index_path = item / "index"
                    metadata_path = item / "metadata.json"
                    logger.info(f"    📂 Index exists: {index_path.exists()}")
                    logger.info(f"    📋 Metadata exists: {metadata_path.exists()}")
                    
        except Exception as e:
            logger.error(f"❌ Cannot read indexed courses: {e}")
    
    logger.info("✅ Simple course directory check completed")

if __name__ == "__main__":
    check_course_directories()