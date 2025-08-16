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
        
        logger.info(f" Offline course manager initialized")
        logger.info(f" Raw docs: {self.raw_docs_dir}")
        logger.info(f" Indexed: {self.indexed_courses_dir}")
    
    def get_available_courses(self) -> List[Dict[str, Any]]:
        """Get list of available courses without complex imports"""
        courses = []
        course_names = set()
        
        logger.info(" OFFLINE: Starting simple course detection...")
        
        try:
            # Check indexed courses first
            indexed_count = 0
            if self.indexed_courses_dir.exists():
                logger.info(" Checking indexed courses...")
                
                try:
                    course_dirs = list(self.indexed_courses_dir.iterdir())
                except (OSError, PermissionError) as e:
                    logger.error(f" Cannot access indexed courses directory: {e}")
                    course_dirs = []
                
                for course_dir in course_dirs:
                    if course_dir.is_dir():
                        course_name = course_dir.name
                        logger.info(f" Found indexed: {course_name}")
                        
                        # Validate course directory accessibility
                        try:
                            # Check if we can actually access this directory
                            index_path = course_dir / "index"
                            metadata_path = course_dir / "metadata.json"
                            
                            # Test directory access
                            accessible = True
                            access_error = None
                            try:
                                list(course_dir.iterdir())
                            except (OSError, PermissionError) as e:
                                access_error = e
                                logger.warning(f" Course directory not accessible: {course_name} - {e}")
                                accessible = False
                            
                            if not accessible:
                                # Mark as inaccessible but still list it
                                course_info = {
                                    'name': course_name,
                                    'status': 'inaccessible',
                                    'document_count': 'Cannot access',
                                    'last_indexed': 'Unknown',
                                    'error': str(access_error) if access_error else 'Directory not accessible'
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
                                    logger.warning(f" Cannot read metadata for {course_name}: {e}")
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
                            logger.error(f" Error processing course {course_name}: {e}")
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
            
            logger.info(f" Found {indexed_count} indexed courses")
            
            # Check raw courses (unprocessed)
            unprocessed_count = 0
            if self.raw_docs_dir.exists():
                logger.info(" Checking raw courses...")
                
                for course_dir in self.raw_docs_dir.iterdir():
                    if course_dir.is_dir() and course_dir.name not in course_names:
                        course_name = course_dir.name
                        logger.info(f" Found unprocessed: {course_name}")
                        
                        # Count documents in the directory
                        doc_count = 0
                        supported_extensions = {'.pdf', '.docx', '.pptx', '.epub', '.txt', '.md'}
                        
                        try:
                            for file_path in course_dir.rglob('*'):
                                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                                    doc_count += 1
                        except Exception as e:
                            logger.warning(f" Cannot count documents in {course_name}: {e}")
                        
                        course_info = {
                            'name': course_name,
                            'status': 'unprocessed',
                            'document_count': doc_count,
                            'raw_directory': str(course_dir),
                        }
                        
                        courses.append(course_info)
                        course_names.add(course_name)
                        unprocessed_count += 1
            
            logger.info(f" Found {unprocessed_count} unprocessed courses")
            logger.info(f" Total courses detected: {len(courses)}")
            
            return courses
        
        except Exception as e:
            logger.error(f" Error in offline course detection: {e}")
            return []
    
    def get_course_analytics(self, course_name):
        """Get analytics for a specific course."""
        # Check if course is indexed
        indexed_path = self.indexed_courses_dir / course_name
        if indexed_path.exists():
            # Count files in indexed course
            file_count = 0
            for file_path in indexed_path.rglob('*'):
                if file_path.is_file():
                    file_count += 1
            
            return {
                'total_documents': file_count,
                'indexed': True,
                'status': 'ready'
            }
        else:
            # Check raw course
            raw_path = self.raw_docs_dir / course_name
            if raw_path.exists():
                file_count = 0
                for file_path in raw_path.rglob('*'):
                    if file_path.is_file():
                        file_count += 1
                
                return {
                    'total_documents': file_count,
                    'indexed': False,
                    'status': 'unprocessed'
                }
        
        return {
            'total_documents': 0,
            'indexed': False,
            'status': 'not_found'
        }
    
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
        
        logger.info(f" Created simple course info for: {course_name}")
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
    
    def index_course_documents(self, course_name: str, documents: List[Dict[str, Any]]):
        """
        Index documents for a specific course in offline mode.
        Creates a simple JSON-based storage without complex dependencies.
        
        Args:
            course_name: Name of the course
            documents: List of processed document dictionaries
        """
        logger.info(f"Indexing {len(documents)} documents for course: {course_name}")
        
        try:
            # Create course directory
            course_index_dir = self.indexed_courses_dir / course_name
            course_index_dir.mkdir(exist_ok=True)
            
            # Save documents in simple JSON format for offline mode
            documents_path = course_index_dir / "documents.json"
            
            # Prepare documents for storage
            stored_docs = []
            for doc in documents:
                stored_doc = {
                    'file_name': doc.get('file_name', ''),
                    'file_type': doc.get('file_type', ''),
                    'file_path': doc.get('file_path', ''),
                    'content': doc.get('content', ''),
                    'is_syllabus': doc.get('is_syllabus', False),
                    'syllabus_weight': doc.get('syllabus_weight', 1.0),
                    'character_count': doc.get('character_count', len(doc.get('content', ''))),
                    'word_count': doc.get('word_count', len(doc.get('content', '').split())),
                    'processed_at': datetime.now().isoformat()
                }
                stored_docs.append(stored_doc)
            
            # Save documents to JSON
            with open(documents_path, 'w', encoding='utf-8') as f:
                json.dump(stored_docs, f, indent=2, ensure_ascii=False)
            
            # Save course metadata
            metadata = {
                'course_name': course_name,
                'document_count': len(documents),
                'syllabus_documents': sum(1 for doc in documents if doc.get('is_syllabus', False)),
                'last_indexed': datetime.now().isoformat(),
                'document_types': self._get_document_type_counts(documents),
                'total_content_length': sum(doc.get('character_count', 0) for doc in documents),
                'storage_format': 'offline_json',
                'indexed_by': 'OfflineCourseManager'
            }
            
            metadata_path = course_index_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully indexed course in offline mode: {course_name}")
            
        except Exception as e:
            logger.error(f"Error indexing course {course_name}: {e}")
            raise
    
    def _get_document_type_counts(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of each document type"""
        type_counts = {}
        for doc in documents:
            doc_type = doc.get('file_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts