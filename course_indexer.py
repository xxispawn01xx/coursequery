"""
Course indexing module using LlamaIndex for document storage and retrieval.
Handles course-specific indexing with syllabus weighting.
"""

# Offline mode compatibility fix - handle pydantic import issues gracefully
try:
    # Try to import pydantic normally first
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Create minimal mock BaseModel for offline mode
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    PYDANTIC_AVAILABLE = False
    print("⚠️ Pydantic not available - using minimal mock for offline mode")

# Handle internal pydantic issues
try:
    import pydantic._internal
    if not hasattr(pydantic._internal, '_model_construction'):
        class MockConstruction:
            pass
        pydantic._internal._model_construction = MockConstruction()
except:
    pass

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pickle

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import Config
from local_models import LocalModelManager

logger = logging.getLogger(__name__)

class CustomEmbeddingWrapper(BaseEmbedding):
    """Wrapper to use our local embedding model with LlamaIndex."""
    
    def __init__(self, model_manager: LocalModelManager, **kwargs):
        """Initialize with local model manager."""
        super().__init__(**kwargs)
        self._model_manager = model_manager
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._model_manager.generate_embeddings([query])[0]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (async version)."""
        return self._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        return self._model_manager.generate_embeddings([text])[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return self._model_manager.generate_embeddings(texts)

class CourseIndexer:
    """Manages indexing and retrieval for course documents."""
    
    def __init__(self):
        """Initialize the course indexer."""
        self.config = Config()
        self.model_manager = None
        self.embedding_wrapper = None
        self.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_config['chunk_size'],
            chunk_overlap=self.config.chunk_config['chunk_overlap'],
        )
        
        # We'll set up our local LLM when model_manager is available
        # No need to disable LLM globally since we want to use our local models
    
    def set_model_manager(self, model_manager: LocalModelManager):
        """Set the model manager for embeddings and LLM."""
        self.model_manager = model_manager
        self.embedding_wrapper = CustomEmbeddingWrapper(model_manager)
        
        # Configure LlamaIndex to use our local models
        try:
            from llama_index.llms.huggingface import HuggingFaceLLM
            
            # Create a local LLM wrapper for LlamaIndex
            local_llm = HuggingFaceLLM(
                model_name="mistralai/Mistral-7B-Instruct-v0.1",
                tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
                context_window=4096,
                max_new_tokens=512,
                generate_kwargs={"temperature": 0.7, "do_sample": True},
                device_map="auto",
            )
            
            # Set global LlamaIndex settings to use our local models
            Settings.llm = local_llm
            Settings.embed_model = self.embedding_wrapper
            
            logger.info("LlamaIndex configured to use local models")
            
        except ImportError:
            logger.warning("HuggingFaceLLM not available, using manual generation")
            # Set embedding model only
            Settings.embed_model = self.embedding_wrapper
    
    def index_course_documents(self, course_name: str, documents: List[Dict[str, Any]]):
        """
        Index documents for a specific course.
        
        Args:
            course_name: Name of the course
            documents: List of processed document dictionaries
        """
        logger.info(f"Indexing {len(documents)} documents for course: {course_name}")
        
        try:
            # Create course directory
            course_index_dir = self.config.indexed_courses_dir / course_name
            course_index_dir.mkdir(exist_ok=True)
            
            # Convert to LlamaIndex documents
            llama_docs = []
            for doc in documents:
                # Create metadata
                metadata = {
                    'file_name': doc['file_name'],
                    'file_type': doc['file_type'],
                    'file_path': doc['file_path'],
                    'is_syllabus': doc['is_syllabus'],
                    'syllabus_weight': doc['syllabus_weight'],
                    'character_count': doc['character_count'],
                    'word_count': doc['word_count'],
                }
                
                # Create LlamaIndex document
                llama_doc = Document(
                    text=doc['content'],
                    metadata=metadata,
                    excluded_llm_metadata_keys=['file_path', 'character_count', 'word_count']
                )
                llama_docs.append(llama_doc)
            
            # Create or update index
            index_path = course_index_dir / "index"
            
            if index_path.exists():
                # Load existing index and add documents
                logger.info(f"Loading existing index for {course_name}")
                storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
                index = load_index_from_storage(storage_context)
                
                # Add new documents
                for doc in llama_docs:
                    index.insert(doc)
            else:
                # Create new index
                logger.info(f"Creating new index for {course_name}")
                
                # Use local embeddings if available
                if self.embedding_wrapper:
                    index = VectorStoreIndex.from_documents(
                        llama_docs,
                        embed_model=self.embedding_wrapper,
                        node_parser=self.node_parser
                    )
                else:
                    # Fallback to HuggingFace embeddings
                    embed_model = HuggingFaceEmbedding(
                        model_name=self.config.model_config['embeddings']['model_name']
                    )
                    index = VectorStoreIndex.from_documents(
                        llama_docs,
                        embed_model=embed_model,
                        node_parser=self.node_parser
                    )
            
            # Persist index
            index.storage_context.persist(persist_dir=str(index_path))
            
            # Save course metadata
            metadata = {
                'course_name': course_name,
                'document_count': len(documents),
                'total_documents': len(llama_docs),
                'syllabus_documents': sum(1 for doc in documents if doc['is_syllabus']),
                'last_indexed': datetime.now().isoformat(),
                'document_types': self._get_document_type_counts(documents),
                'total_content_length': sum(doc['character_count'] for doc in documents),
            }
            
            metadata_path = course_index_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully indexed course: {course_name}")
            
        except Exception as e:
            logger.error(f"Error indexing course {course_name}: {e}")
            raise
    
    def get_course_index(self, course_name: str) -> Optional[VectorStoreIndex]:
        """
        Load index for a specific course with comprehensive debugging.
        
        Args:
            course_name: Name of the course
            
        Returns:
            VectorStoreIndex or None if not found
        """
        try:
            logger.info(f"🔍 DEBUG: Starting index load for course: {course_name}")
            
            course_index_dir = self.config.indexed_courses_dir / course_name
            index_path = course_index_dir / "index"
            
            logger.info(f"📁 DEBUG: Course index directory: {course_index_dir}")
            logger.info(f"📂 DEBUG: Index path: {index_path}")
            logger.info(f"📋 DEBUG: Index path exists: {index_path.exists()}")
            
            if not index_path.exists():
                logger.warning(f"❌ No index found for course: {course_name}")
                return None
            
            # Check directory contents
            if index_path.exists():
                try:
                    contents = list(index_path.iterdir())
                    logger.info(f"📄 DEBUG: Index directory contains {len(contents)} files:")
                    for item in contents[:5]:  # Show first 5 files
                        logger.info(f"  - {item.name}")
                    if len(contents) > 5:
                        logger.info(f"  ... and {len(contents) - 5} more files")
                except Exception as dir_error:
                    logger.error(f"❌ DEBUG: Cannot read index directory: {dir_error}")
            
            logger.info(f"⏳ DEBUG: Creating storage context from {index_path}")
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            logger.info(f"✅ DEBUG: Storage context created successfully")
            
            # Check embedding configuration
            logger.info(f"🤖 DEBUG: Embedding wrapper available: {self.embedding_wrapper is not None}")
            
            # Load with timeout protection
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Index loading timed out after 30 seconds")
            
            # Set timeout for index loading
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                logger.info(f"⏳ DEBUG: Starting index load from storage...")
                start_time = time.time()
                
                # Load with custom embeddings if available
                if self.embedding_wrapper:
                    logger.info(f"🔧 DEBUG: Loading with custom embedding wrapper")
                    index = load_index_from_storage(
                        storage_context,
                        embed_model=self.embedding_wrapper
                    )
                else:
                    logger.info(f"🔧 DEBUG: Loading with default embeddings")
                    index = load_index_from_storage(storage_context)
                
                load_time = time.time() - start_time
                logger.info(f"✅ DEBUG: Index loaded successfully in {load_time:.2f} seconds")
                
            finally:
                signal.alarm(0)  # Cancel timeout
            
            logger.info(f"✅ Loaded index for course: {course_name}")
            return index
            
        except TimeoutError as e:
            logger.error(f"⏰ TIMEOUT: Index loading for course {course_name} timed out after 30 seconds")
            logger.error("💡 SUGGESTION: This course index may be corrupted or too large")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading index for course {course_name}: {e}")
            logger.error(f"🔍 DEBUG: Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"📋 DEBUG: Full traceback:\n{traceback.format_exc()}")
            return None
    
    def get_available_courses(self) -> List[Dict[str, Any]]:
        """
        Get list of available courses with metadata, including both indexed and unprocessed courses.
        
        Returns:
            List of course information dictionaries
        """
        courses = []
        course_names = set()
        
        logger.info("🔍 Starting course detection...")
        logger.info(f"📁 Raw docs directory: {self.config.raw_docs_dir}")
        logger.info(f"📊 Indexed courses directory: {self.config.indexed_courses_dir}")
        
        try:
            # First, get indexed courses from indexed_courses_dir
            indexed_count = 0
            if self.config.indexed_courses_dir.exists():
                logger.info("📊 Checking indexed courses directory...")
                logger.info(f"📁 DEBUG: Indexed courses path: {self.config.indexed_courses_dir}")
                
                try:
                    dirs = list(self.config.indexed_courses_dir.iterdir())
                    logger.info(f"🔍 DEBUG: Found {len(dirs)} items in indexed courses directory")
                except Exception as dir_error:
                    logger.error(f"❌ DEBUG: Cannot read indexed courses directory: {dir_error}")
                    dirs = []
                
                for course_dir in dirs:
                    if course_dir.is_dir():
                        logger.info(f"  📂 Found indexed directory: {course_dir.name}")
                        
                        # Check if this directory actually contains an index
                        index_path = course_dir / "index"
                        metadata_path = course_dir / "metadata.json"
                        
                        logger.info(f"    🔍 Index exists: {index_path.exists()}")
                        logger.info(f"    🔍 Metadata exists: {metadata_path.exists()}")
                        metadata_path = course_dir / "metadata.json"
                        
                        if metadata_path.exists():
                            logger.info(f"    ✅ Has metadata file")
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            course_info = {
                                'name': metadata.get('course_name', course_dir.name),
                                'document_count': metadata.get('document_count', 0),
                                'syllabus_documents': metadata.get('syllabus_documents', 0),
                                'last_indexed': metadata.get('last_indexed', 'Unknown'),
                                'total_content_length': metadata.get('total_content_length', 0),
                                'status': 'indexed',
                                'has_index': True
                            }
                            courses.append(course_info)
                            course_names.add(course_info['name'])
                            indexed_count += 1
                        else:
                            logger.info(f"    ⚠️ Missing metadata file")
                            # Course directory exists but no metadata
                            course_info = {
                                'name': course_dir.name,
                                'document_count': 0,
                                'syllabus_documents': 0,
                                'last_indexed': 'Unknown',
                                'total_content_length': 0,
                                'status': 'indexed',
                                'has_index': True
                            }
                            courses.append(course_info)
                            course_names.add(course_info['name'])
                            indexed_count += 1
                logger.info(f"📊 Found {indexed_count} indexed courses")
            else:
                logger.info("📊 Indexed courses directory does not exist")
            
            # Then, check for unprocessed courses in raw_docs_dir
            unprocessed_count = 0
            if self.config.raw_docs_dir.exists():
                logger.info("📁 Checking raw docs directory...")
                for course_dir in self.config.raw_docs_dir.iterdir():
                    if course_dir.is_dir() and course_dir.name not in course_names:
                        logger.info(f"  Found raw directory: {course_dir.name}")
                        
                        # Count files in the directory
                        supported_files = []
                        all_files = []
                        for file_path in course_dir.rglob('*'):
                            if file_path.is_file():
                                all_files.append(file_path.name)
                                # Check if it's a supported format
                                supported_extensions = ['.pdf', '.docx', '.pptx', '.epub', '.mp4', '.avi', '.mov', '.mp3', '.wav']
                                if file_path.suffix.lower() in supported_extensions:
                                    supported_files.append(file_path)
                        
                        logger.info(f"    All files: {all_files}")
                        logger.info(f"    Supported files: {[f.name for f in supported_files]}")
                        
                        if supported_files:  # Only show directories with supported files
                            course_info = {
                                'name': course_dir.name,
                                'document_count': len(supported_files),
                                'syllabus_documents': 0,
                                'last_indexed': 'Not processed',
                                'total_content_length': 0,
                                'status': 'unprocessed',
                                'has_index': False
                            }
                            courses.append(course_info)
                            unprocessed_count += 1
                            logger.info(f"    ✅ Added as unprocessed course ({len(supported_files)} files)")
                        else:
                            logger.info(f"    ❌ No supported files found")
                logger.info(f"📁 Found {unprocessed_count} unprocessed courses")
            else:
                logger.info("📁 Raw docs directory does not exist")
            
            # Sort: indexed courses first (by last indexed date), then unprocessed courses
            def sort_key(course):
                if course['status'] == 'indexed':
                    return (0, course['last_indexed'])
                else:
                    return (1, course['name'])  # Unprocessed courses sorted by name
            
            courses.sort(key=sort_key, reverse=False)
            
            logger.info(f"🎯 Final result: {len(courses)} total courses")
            for course in courses:
                logger.info(f"  - {course['name']}: {course['status']} ({course['document_count']} docs)")
            
        except Exception as e:
            logger.error(f"❌ Error getting available courses: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return courses
    
    def get_course_analytics(self, course_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed analytics for a specific course.
        
        Args:
            course_name: Name of the course
            
        Returns:
            Analytics dictionary or None if course not found
        """
        try:
            course_index_dir = self.config.indexed_courses_dir / course_name
            metadata_path = course_index_dir / "metadata.json"
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Additional analytics calculations
            analytics = metadata.copy()
            
            # Ensure required fields exist with defaults
            analytics.setdefault('total_documents', 0)
            analytics.setdefault('total_chunks', 0)
            analytics.setdefault('syllabus_documents', 0)
            analytics.setdefault('document_types', {})
            
            # Load raw document data if available
            raw_docs_path = course_index_dir / "raw_documents.pkl"
            if raw_docs_path.exists():
                with open(raw_docs_path, 'rb') as f:
                    raw_docs = pickle.load(f)
                
                # Calculate content length distribution
                content_lengths = [doc['character_count'] for doc in raw_docs if 'character_count' in doc]
                analytics['content_lengths'] = content_lengths
                
                # Calculate average document length
                if content_lengths:
                    analytics['avg_content_length'] = sum(content_lengths) / len(content_lengths)
                    analytics['min_content_length'] = min(content_lengths)
                    analytics['max_content_length'] = max(content_lengths)
                else:
                    analytics['avg_content_length'] = 0
                    analytics['min_content_length'] = 0
                    analytics['max_content_length'] = 0
            
            # Calculate total chunks if we have an index
            try:
                index = self.get_course_index(course_name)
                if index and hasattr(index, 'docstore') and hasattr(index.docstore, 'docs'):
                    analytics['total_chunks'] = len(index.docstore.docs)
                elif index and hasattr(index, '_vector_store') and hasattr(index._vector_store, '_data'):
                    analytics['total_chunks'] = len(index._vector_store._data.embedding_dict)
            except Exception as e:
                logger.warning(f"Could not calculate chunks for {course_name}: {e}")
                # Keep default value of 0
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics for course {course_name}: {e}")
            return None
    
    def reindex_course(self, course_name: str):
        """
        Re-index a course by processing its raw documents again.
        
        Args:
            course_name: Name of the course to re-index
        """
        logger.info(f"Re-indexing course: {course_name}")
        
        try:
            # Load raw documents
            course_raw_dir = self.config.raw_docs_dir / course_name
            
            if not course_raw_dir.exists():
                raise ValueError(f"No raw documents found for course: {course_name}")
            
            # Import document processor
            from document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            
            # Process all files in the course directory
            documents = []
            for file_path in course_raw_dir.iterdir():
                if file_path.is_file() and doc_processor.is_supported_format(file_path):
                    # Assume syllabus files contain 'syllabus' in the name
                    is_syllabus = 'syllabus' in file_path.name.lower()
                    
                    processed_doc = doc_processor.process_file(file_path, is_syllabus)
                    documents.append(processed_doc)
            
            if documents:
                # Remove existing index
                course_index_dir = self.config.indexed_courses_dir / course_name
                if course_index_dir.exists():
                    import shutil
                    shutil.rmtree(course_index_dir)
                
                # Re-index
                self.index_course_documents(course_name, documents)
                logger.info(f"Successfully re-indexed course: {course_name}")
            else:
                raise ValueError(f"No processable documents found for course: {course_name}")
            
        except Exception as e:
            logger.error(f"Error re-indexing course {course_name}: {e}")
            raise
    
    def _get_document_type_counts(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of documents by type."""
        type_counts = {}
        for doc in documents:
            file_type = doc['file_type']
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        return type_counts
    
    def delete_course(self, course_name: str):
        """
        Delete a course and its index.
        
        Args:
            course_name: Name of the course to delete
        """
        try:
            course_index_dir = self.config.indexed_courses_dir / course_name
            
            if course_index_dir.exists():
                import shutil
                shutil.rmtree(course_index_dir)
                logger.info(f"Deleted course: {course_name}")
            else:
                logger.warning(f"Course not found: {course_name}")
                
        except Exception as e:
            logger.error(f"Error deleting course {course_name}: {e}")
            raise
    
    def create_book_index(self, content: str, book_name: str, output_dir: str) -> Dict:
        """Create vector embeddings for individual book content."""
        try:
            from llama_index.core import Document, VectorStoreIndex, Settings
            from llama_index.core.text_splitter import TokenTextSplitter
            import json
            
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Split into chunks
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(content)
            
            # Create documents for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    metadata={
                        'source': book_name,
                        'chunk_id': i,
                        'book_name': book_name
                    }
                )
                documents.append(doc)
            
            # Create index with our local embedding model if available
            if self.embedding_wrapper:
                Settings.embed_model = self.embedding_wrapper
            
            index = VectorStoreIndex.from_documents(documents)
            
            # Save index
            index.storage_context.persist(persist_dir=str(output_path))
            
            # Save metadata
            metadata = {
                'book_name': book_name,
                'chunk_count': len(chunks),
                'created_date': datetime.now().isoformat(),
                'model_name': 'local_embeddings' if self.embedding_wrapper else 'default',
                'content_length': len(content)
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created book index for '{book_name}' with {len(chunks)} chunks")
            
            return {
                'success': True,
                'chunk_count': len(chunks),
                'model_name': 'local_embeddings' if self.embedding_wrapper else 'default',
                'output_dir': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Error creating book index for '{book_name}': {e}")
            return {
                'success': False,
                'error': str(e)
            }
