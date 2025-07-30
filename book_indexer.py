"""
Book Indexer - Individual book/ebook embedding system
Processes individual books within course directories for granular vector embeddings
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class BookIndexer:
    """Manages individual book indexing within course directories."""
    
    def __init__(self, config_dir: str = "."):
        self.config_dir = Path(config_dir)
        self.book_index_file = self.config_dir / "book_embeddings.json"
        self.embeddings_dir = self.config_dir / "book_embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Book-specific supported formats
        self.book_extensions = {'.pdf', '.epub', '.docx', '.txt', '.md'}
        self.load_book_index()
    
    def load_book_index(self) -> None:
        """Load book embedding index from storage."""
        try:
            if self.book_index_file.exists():
                with open(self.book_index_file, 'r', encoding='utf-8') as f:
                    self.book_index = json.load(f)
                logger.info(f"Loaded book index with {len(self.book_index)} entries")
            else:
                self.book_index = {}
                logger.info("No book index found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading book index: {e}")
            self.book_index = {}
    
    def save_book_index(self) -> None:
        """Save book embedding index to storage."""
        try:
            with open(self.book_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.book_index, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved book index with {len(self.book_index)} entries")
        except Exception as e:
            logger.error(f"Error saving book index: {e}")
    
    def scan_directory_for_books(self, directory_path: Path) -> List[Dict]:
        """Scan directory and return list of individual books found."""
        books = []
        
        try:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.book_extensions:
                    # Generate unique book ID
                    book_id = self.generate_book_id(file_path)
                    
                    # Check if already indexed
                    is_indexed = book_id in self.book_index
                    
                    # Get file stats
                    file_size = file_path.stat().st_size
                    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    book_info = {
                        'id': book_id,
                        'name': file_path.stem,
                        'full_name': file_path.name,
                        'path': str(file_path),
                        'relative_path': str(file_path.relative_to(directory_path)),
                        'extension': file_path.suffix.lower(),
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'modified': modified_time.isoformat(),
                        'is_indexed': is_indexed,
                        'parent_course': directory_path.name
                    }
                    
                    # Add indexing info if available
                    if is_indexed:
                        index_info = self.book_index[book_id]
                        book_info['indexed_date'] = index_info.get('indexed_date')
                        book_info['chunk_count'] = index_info.get('chunk_count', 0)
                        book_info['embedding_model'] = index_info.get('embedding_model', 'unknown')
                    
                    books.append(book_info)
        
        except Exception as e:
            logger.error(f"Error scanning directory for books: {e}")
        
        return sorted(books, key=lambda x: x['name'])
    
    def generate_book_id(self, file_path: Path) -> str:
        """Generate unique ID for a book based on path and content hash."""
        # Use file path and size for ID generation
        path_str = str(file_path)
        try:
            file_size = file_path.stat().st_size
            content_hash = hashlib.md5(f"{path_str}_{file_size}".encode()).hexdigest()[:8]
            return f"book_{content_hash}"
        except Exception:
            return f"book_{hashlib.md5(path_str.encode()).hexdigest()[:8]}"
    
    def process_book(self, book_info: Dict, document_processor=None, course_indexer=None) -> Dict:
        """Process individual book for embedding generation."""
        try:
            book_id = book_info['id']
            file_path = Path(book_info['path'])
            
            if not file_path.exists():
                return {'success': False, 'error': f"File not found: {file_path}"}
            
            logger.info(f"Processing book: {book_info['name']}")
            
            # Process document content
            if document_processor:
                content = document_processor.process_file(str(file_path))
                if not content:
                    return {'success': False, 'error': 'Failed to extract content'}
            else:
                return {'success': False, 'error': 'Document processor not available'}
            
            # Create book-specific index
            if course_indexer:
                # Create embedding directory for this book
                book_embedding_dir = self.embeddings_dir / book_id
                book_embedding_dir.mkdir(exist_ok=True)
                
                # Index the book content
                index_result = course_indexer.create_book_index(
                    content, 
                    book_info['name'],
                    str(book_embedding_dir)
                )
                
                if index_result.get('success'):
                    # Update book index
                    self.book_index[book_id] = {
                        'name': book_info['name'],
                        'path': book_info['path'],
                        'indexed_date': datetime.now().isoformat(),
                        'chunk_count': index_result.get('chunk_count', 0),
                        'embedding_model': index_result.get('model_name', 'unknown'),
                        'embedding_dir': str(book_embedding_dir),
                        'parent_course': book_info['parent_course']
                    }
                    
                    self.save_book_index()
                    
                    return {
                        'success': True,
                        'chunk_count': index_result.get('chunk_count', 0),
                        'embedding_dir': str(book_embedding_dir)
                    }
                else:
                    return {'success': False, 'error': 'Failed to create embeddings'}
            else:
                return {'success': False, 'error': 'Course indexer not available'}
                
        except Exception as e:
            logger.error(f"Error processing book {book_info['name']}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_book_stats(self) -> Dict:
        """Get statistics about indexed books."""
        total_indexed = len(self.book_index)
        total_chunks = sum(info.get('chunk_count', 0) for info in self.book_index.values())
        
        # Group by parent course
        courses = {}
        for book_info in self.book_index.values():
            course = book_info.get('parent_course', 'unknown')
            if course not in courses:
                courses[course] = {'books': 0, 'chunks': 0}
            courses[course]['books'] += 1
            courses[course]['chunks'] += book_info.get('chunk_count', 0)
        
        return {
            'total_books': total_indexed,
            'total_chunks': total_chunks,
            'courses': courses,
            'embedding_dir': str(self.embeddings_dir),
            'index_file': str(self.book_index_file)
        }
    
    def search_books(self, query: str, course_filter: Optional[str] = None) -> List[Dict]:
        """Search for books by name or content."""
        results = []
        
        for book_id, book_info in self.book_index.items():
            # Apply course filter if specified
            if course_filter and book_info.get('parent_course') != course_filter:
                continue
            
            # Simple text matching (can be enhanced with embeddings)
            if query.lower() in book_info['name'].lower():
                results.append({
                    'id': book_id,
                    'name': book_info['name'],
                    'course': book_info.get('parent_course'),
                    'chunks': book_info.get('chunk_count', 0),
                    'indexed_date': book_info.get('indexed_date')
                })
        
        return sorted(results, key=lambda x: x['name'])
    
    def delete_book_embedding(self, book_id: str) -> bool:
        """Delete book embedding and index entry."""
        try:
            if book_id in self.book_index:
                # Remove embedding directory
                embedding_dir = Path(self.book_index[book_id].get('embedding_dir', ''))
                if embedding_dir.exists():
                    import shutil
                    shutil.rmtree(embedding_dir)
                
                # Remove from index
                del self.book_index[book_id]
                self.save_book_index()
                
                logger.info(f"Deleted book embedding: {book_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting book embedding {book_id}: {e}")
            return False