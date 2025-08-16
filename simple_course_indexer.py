"""
Simple course indexer that works without complex LlamaIndex dependencies
Provides basic course indexing and document counting for offline mode
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from document_processor import DocumentProcessor
from directory_config import get_master_directory

logger = logging.getLogger(__name__)

class SimpleCourseIndexer:
    """Simple course indexer that works without LlamaIndex complex dependencies"""
    
    def __init__(self):
        """Initialize simple indexer"""
        self.master_dir = Path(get_master_directory())
        self.indexed_dir = Path("indexed_courses")
        self.indexed_dir.mkdir(exist_ok=True)
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor()
        
        logger.info(f" Simple indexer initialized")
        logger.info(f" Source: {self.master_dir}")
        logger.info(f" Indexed: {self.indexed_dir}")
    
    def process_course_simple(self, course_name: str) -> Dict[str, Any]:
        """
        Process a course and create simple index with document counting
        
        Args:
            course_name: Name of the course directory
            
        Returns:
            Course processing results
        """
        logger.info(f" Processing course: {course_name}")
        
        course_path = self.master_dir / course_name
        if not course_path.exists():
            raise ValueError(f"Course directory not found: {course_path}")
        
        # Create indexed course directory
        indexed_course_dir = self.indexed_dir / course_name
        indexed_course_dir.mkdir(exist_ok=True)
        
        # Process all documents in the course
        all_files = []
        processed_docs = []
        document_types = {}
        total_chars = 0
        
        # Walk through all files in course directory
        for root, dirs, files in os.walk(course_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in ['.pdf', '.docx', '.pptx', '.vtt', '.srt', '.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.csv', '.mp4', '.mp3', '.wav', '.m4a']:
                    all_files.append(file_path)
        
        logger.info(f" Found {len(all_files)} files to process")
        
        # Process each file
        for file_path in all_files:
            try:
                # Use document processor to extract content
                # Pass as Path object, not string
                content = self.doc_processor.process_file(file_path)
                
                if content and len(content.strip()) > 0:
                    doc_info = {
                        'filename': file_path.name,
                        'relative_path': str(file_path.relative_to(course_path)),
                        'content': content,
                        'character_count': len(content),
                        'file_type': file_path.suffix.lower(),
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    processed_docs.append(doc_info)
                    total_chars += len(content)
                    
                    # Count document types
                    file_type = file_path.suffix.lower()
                    document_types[file_type] = document_types.get(file_type, 0) + 1
                    
                    logger.info(f" Processed: {file_path.name} ({len(content)} chars)")
                else:
                    logger.warning(f" No content extracted from: {file_path.name}")
                    
            except Exception as e:
                logger.error(f" Error processing {file_path.name}: {e}")
                continue
        
        # Save processed documents
        docs_file = indexed_course_dir / "documents.pkl"
        with open(docs_file, 'wb') as f:
            pickle.dump(processed_docs, f)
        
        # Create metadata
        metadata = {
            'course_name': course_name,
            'document_count': len(processed_docs),
            'total_files_found': len(all_files),
            'document_types': document_types,
            'total_content_length': total_chars,
            'last_indexed': datetime.now().isoformat(),
            'indexer_type': 'simple',
            'processing_summary': {
                'successful': len(processed_docs),
                'failed': len(all_files) - len(processed_docs),
                'total_attempted': len(all_files)
            }
        }
        
        # Save metadata
        metadata_file = indexed_course_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Course '{course_name}' indexed: {len(processed_docs)} documents, {total_chars} characters")
        
        return metadata
    
    def get_indexed_courses(self) -> List[Dict[str, Any]]:
        """Get list of indexed courses with metadata"""
        courses = []
        
        if not self.indexed_dir.exists():
            return courses
        
        for course_dir in self.indexed_dir.iterdir():
            if course_dir.is_dir():
                metadata_file = course_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        courses.append(metadata)
                    except Exception as e:
                        logger.error(f"Error reading metadata for {course_dir.name}: {e}")
                        # Add basic info even if metadata is corrupted
                        courses.append({
                            'course_name': course_dir.name,
                            'document_count': 'Unknown',
                            'last_indexed': 'Unknown',
                            'status': 'metadata_error'
                        })
        
        return courses
    
    def get_available_raw_courses(self) -> List[str]:
        """Get list of available course directories that can be processed"""
        if not self.master_dir.exists():
            return []
        
        raw_courses = []
        for item in self.master_dir.iterdir():
            if item.is_dir():
                raw_courses.append(item.name)
        
        return raw_courses

def test_simple_indexer():
    """Test the simple indexer"""
    indexer = SimpleCourseIndexer()
    
    # Get available courses
    raw_courses = indexer.get_available_raw_courses()
    indexed_courses = indexer.get_indexed_courses()
    
    print(f"Raw courses available: {len(raw_courses)}")
    for course in raw_courses[:3]:
        print(f"  - {course}")
    
    print(f"Indexed courses: {len(indexed_courses)}")
    for course in indexed_courses:
        print(f"  - {course['course_name']}: {course['document_count']} docs")

if __name__ == "__main__":
    test_simple_indexer()