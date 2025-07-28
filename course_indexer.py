"""
Course indexing module using LlamaIndex for document storage and retrieval.
Handles course-specific indexing with syllabus weighting.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pickle

# LlamaIndex imports with fallbacks for Replit environment
try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    # Mock classes for development environment
    LLAMA_INDEX_AVAILABLE = False
    
    class Document:
        def __init__(self, text="", metadata=None, excluded_llm_metadata_keys=None):
            self.text = text
            self.metadata = metadata or {}
    
    class VectorStoreIndex:
        def __init__(self, documents=None, embed_model=None):
            self.documents = documents or []
        
        def insert(self, document):
            self.documents.append(document)
    
    class StorageContext:
        @staticmethod
        def from_defaults(persist_dir=None):
            return StorageContext()
    
    def load_index_from_storage(storage_context):
        return VectorStoreIndex()
    
    class Settings:
        llm = None
        embed_model = None
    
    class SentenceSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
    
    class BaseEmbedding:
        def __init__(self, **kwargs):
            pass
    
    class HuggingFaceEmbedding:
        def __init__(self, model_name="", **kwargs):
            self.model_name = model_name

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
        Load index for a specific course.
        
        Args:
            course_name: Name of the course
            
        Returns:
            VectorStoreIndex or None if not found
        """
        try:
            course_index_dir = self.config.indexed_courses_dir / course_name
            index_path = course_index_dir / "index"
            
            if not index_path.exists():
                logger.warning(f"No index found for course: {course_name}")
                return None
            
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            
            # Load with custom embeddings if available
            if self.embedding_wrapper:
                index = load_index_from_storage(
                    storage_context,
                    embed_model=self.embedding_wrapper
                )
            else:
                index = load_index_from_storage(storage_context)
            
            logger.info(f"Loaded index for course: {course_name}")
            return index
            
        except Exception as e:
            logger.error(f"Error loading index for course {course_name}: {e}")
            return None
    
    def get_available_courses(self) -> List[Dict[str, Any]]:
        """
        Get list of available courses with metadata.
        
        Returns:
            List of course information dictionaries
        """
        courses = []
        
        try:
            for course_dir in self.config.indexed_courses_dir.iterdir():
                if course_dir.is_dir():
                    metadata_path = course_dir / "metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        course_info = {
                            'name': metadata.get('course_name', course_dir.name),
                            'document_count': metadata.get('document_count', 0),
                            'syllabus_documents': metadata.get('syllabus_documents', 0),
                            'last_indexed': metadata.get('last_indexed', 'Unknown'),
                            'total_content_length': metadata.get('total_content_length', 0),
                        }
                        courses.append(course_info)
                    else:
                        # Course directory exists but no metadata
                        course_info = {
                            'name': course_dir.name,
                            'document_count': 0,
                            'syllabus_documents': 0,
                            'last_indexed': 'Unknown',
                            'total_content_length': 0,
                        }
                        courses.append(course_info)
            
            # Sort by last indexed date
            courses.sort(key=lambda x: x['last_indexed'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting available courses: {e}")
        
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
            
            # Load raw document data if available
            raw_docs_path = course_index_dir / "raw_documents.pkl"
            if raw_docs_path.exists():
                with open(raw_docs_path, 'rb') as f:
                    raw_docs = pickle.load(f)
                
                # Calculate content length distribution
                content_lengths = [doc['character_count'] for doc in raw_docs]
                analytics['content_lengths'] = content_lengths
                
                # Calculate average document length
                if content_lengths:
                    analytics['avg_content_length'] = sum(content_lengths) / len(content_lengths)
                    analytics['min_content_length'] = min(content_lengths)
                    analytics['max_content_length'] = max(content_lengths)
            
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
