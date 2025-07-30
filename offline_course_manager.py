"""
Simple offline course manager that works without complex dependencies
Provides basic course detection and management for pure offline mode
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OfflineCourseManager:
    """Simple course manager that works without LlamaIndex or complex dependencies"""
    
    def __init__(self, raw_docs_dir: Path, indexed_courses_dir: Path):
        """Initialize with directory paths"""
        self.raw_docs_dir = Path(raw_docs_dir)
        self.indexed_courses_dir = Path(indexed_courses_dir)
        
        # Create directories if they don't exist
        self.raw_docs_dir.mkdir(exist_ok=True)
        self.indexed_courses_dir.mkdir(exist_ok=True)
        
        logger.info(f"📁 Offline course manager initialized")
        logger.info(f"📚 Raw docs: {self.raw_docs_dir}")
        logger.info(f"📊 Indexed: {self.indexed_courses_dir}")
    
    def get_available_courses(self) -> List[Dict[str, Any]]:
        """Get list of available courses without complex imports"""
        courses = []
        course_names = set()
        
        logger.info("🔍 OFFLINE: Starting simple course detection...")
        
        try:
            # Check indexed courses first
            indexed_count = 0
            if self.indexed_courses_dir.exists():
                logger.info("📊 Checking indexed courses...")
                
                try:
                    course_dirs = list(self.indexed_courses_dir.iterdir())
                except (OSError, PermissionError) as e:
                    logger.error(f"❌ Cannot access indexed courses directory: {e}")
                    course_dirs = []
                
                for course_dir in course_dirs:
                    if course_dir.is_dir():
                        course_name = course_dir.name
                        logger.info(f"  📂 Found indexed: {course_name}")
                        
                        # Validate course directory accessibility
                        try:
                            # Check if we can actually access this directory
                            index_path = course_dir / "index"
                            metadata_path = course_dir / "metadata.json"
                            
                            # Test directory access
                            accessible = True
                            try:
                                list(course_dir.iterdir())
                            except (OSError, PermissionError) as access_error:
                                logger.warning(f"⚠️ Course directory not accessible: {course_name} - {access_error}")
                                accessible = False
                            
                            if not accessible:
                                # Mark as inaccessible but still list it
                                course_info = {
                                    'name': course_name,
                                    'status': 'inaccessible',
                                    'document_count': 'Cannot access',
                                    'last_indexed': 'Unknown',
                                    'error': str(access_error) if 'access_error' in locals() else 'Directory not accessible'
                                }
                            elif metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r', encoding='utf-8') as f:
                                        metadata = json.load(f)
                                    
                                    course_info = {
                                        'name': course_name,
                                        'status': 'indexed',
                                        'document_count': metadata.get('document_count', 0),
                                        'last_indexed': metadata.get('last_indexed', 'Unknown'),
                                        'total_content_length': metadata.get('total_content_length', 0),
                                        'document_types': metadata.get('document_types', {}),
                                    }
                                    
                                except Exception as e:
                                    logger.warning(f"⚠️ Cannot read metadata for {course_name}: {e}")
                                    course_info = {
                                        'name': course_name,
                                        'status': 'indexed_no_metadata',
                                        'document_count': 'Unknown',
                                        'last_indexed': 'Unknown',
                                        'error': f'Metadata read error: {e}'
                                    }
                            else:
                                course_info = {
                                    'name': course_name,
                                    'status': 'indexed_no_metadata',
                                    'document_count': 'Unknown',
                                    'last_indexed': 'Unknown',
                                }
                            
                            courses.append(course_info)
                            course_names.add(course_name)
                            indexed_count += 1
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing course {course_name}: {e}")
                            # Still add it to the list but mark as problematic
                            courses.append({
                                'name': course_name,
                                'status': 'error',
                                'document_count': 'Error',
                                'last_indexed': 'Unknown',
                                'error': str(e)
                            })
                            course_names.add(course_name)
                            indexed_count += 1
            
            logger.info(f"✅ Found {indexed_count} indexed courses")
            
            # Check raw courses (unprocessed)
            unprocessed_count = 0
            if self.raw_docs_dir.exists():
                logger.info("📚 Checking raw courses...")
                
                for course_dir in self.raw_docs_dir.iterdir():
                    if course_dir.is_dir() and course_dir.name not in course_names:
                        course_name = course_dir.name
                        logger.info(f"  📁 Found unprocessed: {course_name}")
                        
                        # Count documents in the directory
                        doc_count = 0
                        supported_extensions = {'.pdf', '.docx', '.pptx', '.epub', '.txt', '.md'}
                        
                        try:
                            for file_path in course_dir.rglob('*'):
                                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                                    doc_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Cannot count documents in {course_name}: {e}")
                        
                        course_info = {
                            'name': course_name,
                            'status': 'unprocessed',
                            'document_count': doc_count,
                            'raw_directory': str(course_dir),
                        }
                        
                        courses.append(course_info)
                        course_names.add(course_name)
                        unprocessed_count += 1
            
            logger.info(f"✅ Found {unprocessed_count} unprocessed courses")
            logger.info(f"📊 Total courses detected: {len(courses)}")
            
            return courses
            
        except Exception as e:
            logger.error(f"❌ Error in offline course detection: {e}")
            return []
    
    def create_simple_course_info(self, course_name: str, documents: List[Dict]) -> Dict[str, Any]:
        """Create simple course info without complex indexing"""
        course_dir = self.indexed_courses_dir / course_name
        course_dir.mkdir(exist_ok=True)
        
        # Create simple metadata
        metadata = {
            'course_name': course_name,
            'document_count': len(documents),
            'last_indexed': datetime.now().isoformat(),
            'indexed_with': 'offline_course_manager',
            'documents': [
                {
                    'filename': doc.get('filename', 'unknown'),
                    'file_type': doc.get('file_type', 'unknown'),
                    'character_count': doc.get('character_count', 0),
                } for doc in documents
            ]
        }
        
        metadata_path = course_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Created simple course info for: {course_name}")
        return metadata
    
    def get_course_status(self, course_name: str) -> str:
        """Check if a course is indexed or unprocessed"""
        indexed_path = self.indexed_courses_dir / course_name
        raw_path = self.raw_docs_dir / course_name
        
        if indexed_path.exists() and (indexed_path / "metadata.json").exists():
            return "indexed"
        elif raw_path.exists():
            return "unprocessed"
        else:
            return "not_found"
    
    def is_available(self) -> bool:
        """Check if offline course manager is available and working"""
        try:
            self.get_available_courses()
            return True
        except Exception:
            return False