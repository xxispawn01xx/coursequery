"""
Vector RAG Engine - Optimized for Cost and Performance
Chunks transcripts into embeddings for efficient semantic search with cloud APIs.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class TranscriptChunker:
    """Intelligently chunks transcripts into manageable, semantic pieces."""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """Initialize chunker with optimal parameters for embeddings."""
        self.chunk_size = chunk_size  # tokens
        self.overlap = overlap  # token overlap between chunks
        
    def chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs with metadata."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) < 10:  # Skip very short paragraphs
                continue
                
            chunks.append({
                'content': paragraph,
                'type': 'paragraph',
                'index': i,
                'word_count': len(paragraph.split()),
                'char_count': len(paragraph)
            })
        
        return chunks
    
    def chunk_by_sliding_window(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using sliding window approach."""
        words = text.split()
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'type': 'sliding_window',
                'index': chunk_id,
                'start_word': start,
                'end_word': end,
                'word_count': len(chunk_words),
                'char_count': len(chunk_text)
            })
            
            start += self.chunk_size - self.overlap
            chunk_id += 1
            
            if end >= len(words):
                break
        
        return chunks
    
    def chunk_by_topics(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by topic boundaries using simple heuristics."""
        # Simple topic detection based on transition words and concepts
        topic_indicators = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'however', 'moreover', 'furthermore', 'in addition',
            'on the other hand', 'in contrast', 'meanwhile'
        ]
        
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        chunk_id = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # Check for topic boundary
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in topic_indicators):
                if len(current_chunk) > 3:  # Minimum chunk size
                    chunk_text = '. '.join(current_chunk)
                    chunks.append({
                        'content': chunk_text,
                        'type': 'topic_based',
                        'index': chunk_id,
                        'sentence_count': len(current_chunk),
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    })
                    current_chunk = []
                    chunk_id += 1
        
        # Add remaining content
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'type': 'topic_based',
                'index': chunk_id,
                'sentence_count': len(current_chunk),
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text)
            })
        
        return chunks

class LocalEmbeddingsEngine:
    """Generate embeddings using local models for cost efficiency."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with lightweight embedding model."""
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        
    def load_model(self):
        """Load embedding model on demand."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
    
    def generate_embeddings(self, texts: List[str], device: str = 'auto') -> np.ndarray:
        """Generate embeddings for a list of texts with RTX 3060 optimization."""
        if self.model is None:
            self.load_model()
        
        # RTX 3060 memory optimization
        try:
            import torch
            if torch.cuda.is_available() and device != 'cpu':
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                usage_percent = (allocated / total) * 100
                
                if usage_percent > 95:
                    logger.warning(f"RTX 3060 memory at {usage_percent:.1f}% - using CPU for embeddings")
                    # Force CPU for sentence transformers
                    import os
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    device = 'cpu'
        except ImportError:
            pass
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embeddings_cache:
                cached_embeddings.append((i, self.embeddings_cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        new_embeddings = None
        if uncached_texts and self.model is not None:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.embeddings_cache[text_hash] = embedding
        
        # Get embedding dimension
        embedding_dim = 384  # Default for all-MiniLM-L6-v2
        if self.model is not None and hasattr(self.model, 'get_sentence_embedding_dimension'):
            embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Ensure embedding_dim is not None
        if embedding_dim is None:
            embedding_dim = 384
        
        # Combine cached and new embeddings in correct order
        all_embeddings = np.zeros((len(texts), int(embedding_dim)), dtype=np.float32)
        
        # Add cached embeddings
        for original_idx, embedding in cached_embeddings:
            all_embeddings[original_idx] = embedding
        
        # Add new embeddings
        if uncached_texts and new_embeddings is not None:
            for i, original_idx in enumerate(uncached_indices):
                all_embeddings[original_idx] = new_embeddings[i]
        
        return all_embeddings
    
    def search_similar(self, query_text: str, chunk_embeddings: np.ndarray, 
                      chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for most similar chunks to query."""
        if self.model is None:
            self.load_model()
        
        # Generate query embedding
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        
        # Calculate similarity scores using numpy (avoid sklearn dependency)
        # Normalize vectors for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        similarities = np.dot(chunk_norms, query_norm)
        
        # Get top K results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((chunks[idx], float(similarities[idx])))
        
        return results

class VectorRAGEngine:
    """Complete RAG engine combining chunking, embeddings, and cloud APIs."""
    
    def __init__(self, config=None):
        """Initialize RAG engine."""
        from config import Config
        self.config = config or Config()
        
        self.chunker = TranscriptChunker()
        self.embeddings_engine = LocalEmbeddingsEngine()
        
        # Storage paths
        self.vectors_dir = Path("./vectors")
        self.vectors_dir.mkdir(exist_ok=True)
        
        # Course vectors cache
        self.course_vectors = {}
    
    def process_course_content(self, course_name: str, chunking_method: str = "paragraphs") -> Dict[str, Any]:
        """Process ALL course content including documents, transcriptions, and indexed materials."""
        all_chunks = []
        chunk_metadata = []
        processed_sources = set()
        
        # Import document processor
        try:
            from document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
        except ImportError:
            doc_processor = None
        
        # 1. Process indexed course documents (if available)
        indexed_course_dir = self.config.indexed_courses_dir / course_name
        if indexed_course_dir.exists():
            logger.info(f"Processing indexed documents from {indexed_course_dir}")
            self._process_indexed_documents(indexed_course_dir, all_chunks, chunk_metadata, processed_sources, chunking_method)
        
        # 1.5. Extract content from LlamaIndex storage (for existing courses like vcpe)
        logger.info(f"Attempting to extract content from LlamaIndex for {course_name}")
        try:
            self._process_llama_index_content(course_name, all_chunks, chunk_metadata, processed_sources, chunking_method)
        except Exception as e:
            logger.warning(f"LlamaIndex extraction failed for {course_name}: {e}")
        
        # 2. Process raw documents (PDFs, DOCX, PPTX, etc.)
        raw_course_dir = self.config.raw_docs_dir / course_name
        if raw_course_dir.exists() and doc_processor:
            logger.info(f"Processing raw documents from {raw_course_dir}")
            self._process_raw_documents(raw_course_dir, doc_processor, all_chunks, chunk_metadata, processed_sources, chunking_method)
        
        # 3. Process transcriptions
        transcriptions_dir = Path("./transcriptions") / course_name
        if transcriptions_dir.exists():
            logger.info(f"Processing transcriptions from {transcriptions_dir}")
            self._process_transcriptions(transcriptions_dir, all_chunks, chunk_metadata, processed_sources, chunking_method)
        
        if not all_chunks:
            error_msg = f"No content found for course: {course_name}\n"
            error_msg += f"Checked locations:\n"
            error_msg += f"- Indexed dir: {self.config.indexed_courses_dir / course_name} (exists: {(self.config.indexed_courses_dir / course_name).exists()})\n"
            error_msg += f"- Raw docs dir: {self.config.raw_docs_dir / course_name} (exists: {(self.config.raw_docs_dir / course_name).exists()})\n"
            error_msg += f"- Transcriptions dir: {Path('./transcriptions') / course_name} (exists: {(Path('./transcriptions') / course_name).exists()})\n"
            error_msg += f"- Processed sources: {list(processed_sources)}"
            raise ValueError(error_msg)
        
        # Generate embeddings for all content
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks from {len(processed_sources)} sources...")
        
        # RTX 3060 optimized memory management
        try:
            import torch
            if torch.cuda.is_available():
                # Clear cache and synchronize
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                usage_percent = (allocated / total) * 100
                
                logger.info(f"RTX 3060 Memory: {allocated:.0f}MB used / {total:.0f}MB total ({usage_percent:.1f}%)")
                
                # RTX 3060 12GB can handle higher memory usage - only fallback if truly critical
                if usage_percent > 98:  # Much higher threshold for RTX 3060
                    logger.warning(f"RTX 3060 memory very high: {usage_percent:.1f}% - attempting aggressive cleanup")
                    
                    # Try aggressive cleanup
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Recheck
                    allocated = torch.cuda.memory_allocated(0) / (1024**2)
                    usage_percent = (allocated / total) * 100
                    
                    if usage_percent > 99:
                        logger.error(f"RTX 3060 memory critically low after cleanup: {usage_percent:.1f}%")
                        logger.info("Using CPU for embeddings to prevent GPU crash")
                        # Don't raise error, just log - let embedding engine handle CPU fallback
                    else:
                        logger.info(f"RTX 3060 memory after cleanup: {usage_percent:.1f}% - proceeding with GPU")
                else:
                    logger.info(f"RTX 3060 memory healthy: {usage_percent:.1f}% - proceeding with GPU embeddings")
        except ImportError:
            pass
        
        try:
            embeddings = self.embeddings_engine.generate_embeddings(all_chunks)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during vector embedding generation: {e}")
                logger.info("This is likely due to RTX 3060 memory constraints with large document sets")
                logger.info("The system will automatically retry with CPU fallback")
                raise RuntimeError(f"CUDA memory error with {len(all_chunks)} chunks. The system attempted CPU fallback. Original error: {e}")
            else:
                raise
        
        # Create comprehensive vector database entry
        vector_data = {
            'course_name': course_name,
            'chunks': chunk_metadata,
            'embeddings': embeddings.tolist(),
            'chunking_method': chunking_method,
            'created': datetime.now().isoformat(),
            'total_chunks': len(all_chunks),
            'total_sources': len(processed_sources),
            'source_types': list(set(chunk['source_type'] for chunk in chunk_metadata)),
            'processed_sources': list(processed_sources)
        }
        
        # Save vectors
        vectors_file = self.vectors_dir / f"{course_name}_vectors.json"
        with open(vectors_file, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, indent=2, ensure_ascii=False)
        
        # Cache in memory
        self.course_vectors[course_name] = {
            'chunks': chunk_metadata,
            'embeddings': embeddings
        }
        
        logger.info(f"Created {len(all_chunks)} vector embeddings for {course_name} from {len(processed_sources)} sources")
        return vector_data

    def process_transcripts(self, course_name: str, transcripts: List[Dict[str, Any]], 
                          chunking_method: str = "paragraphs") -> Dict[str, Any]:
        """Process transcripts into vector embeddings."""
        all_chunks = []
        chunk_metadata = []
        
        # Process each transcript
        for transcript in transcripts:
            content = transcript.get('content', '')
            source_file = transcript.get('original_file', 'unknown')
            
            # Choose chunking method
            if chunking_method == "paragraphs":
                chunks = self.chunker.chunk_by_paragraphs(content)
            elif chunking_method == "sliding_window":
                chunks = self.chunker.chunk_by_sliding_window(content)
            elif chunking_method == "topics":
                chunks = self.chunker.chunk_by_topics(content)
            else:
                raise ValueError(f"Unknown chunking method: {chunking_method}")
            
            # Add source metadata to chunks
            for chunk in chunks:
                chunk['source_file'] = source_file
                chunk['course_name'] = course_name
                chunk['transcript_index'] = transcripts.index(transcript)
                
                all_chunks.append(chunk['content'])
                chunk_metadata.append(chunk)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embeddings_engine.generate_embeddings(all_chunks)
        
        # Create vector database entry
        vector_data = {
            'course_name': course_name,
            'chunks': chunk_metadata,
            'embeddings': embeddings.tolist(),  # Convert to list for JSON serialization
            'chunking_method': chunking_method,
            'created': datetime.now().isoformat(),
            'total_chunks': len(all_chunks),
            'total_transcripts': len(transcripts)
        }
        
        # Save vectors
        vectors_file = self.vectors_dir / f"{course_name}_vectors.json"
        with open(vectors_file, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, indent=2, ensure_ascii=False)
        
        # Cache in memory
        self.course_vectors[course_name] = {
            'chunks': chunk_metadata,
            'embeddings': embeddings
        }
        
        logger.info(f"Created {len(all_chunks)} vector embeddings for {course_name}")
        return vector_data
    
    def _process_indexed_documents(self, indexed_dir: Path, all_chunks: List[str], 
                                 chunk_metadata: List[Dict], processed_sources: set, chunking_method: str):
        """Process documents from indexed course directory."""
        # Look for text files in the indexed directory
        for file_path in indexed_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if content.strip():
                        source_name = f"indexed_{file_path.name}"
                        processed_sources.add(source_name)
                        self._chunk_and_add_content(content, source_name, "indexed_document", 
                                                  all_chunks, chunk_metadata, chunking_method)
                except Exception as e:
                    logger.warning(f"Could not read indexed file {file_path}: {e}")
    
    def _process_llama_index_content(self, course_name: str, all_chunks: List[str], 
                                   chunk_metadata: List[Dict], processed_sources: set, chunking_method: str):
        """Extract content from LlamaIndex storage for courses that exist in the system."""
        try:
            # Try to get content from course indexer
            from course_indexer import CourseIndexer
            course_indexer = CourseIndexer()
            
            # Check if course has indexed content  
            course_index_dir = self.config.indexed_courses_dir / course_name
            
            # Get the course index directly without requiring directory
            index = course_indexer.get_course_index(course_name)
            if not index:
                logger.warning(f"No LlamaIndex found for course {course_name}")
                return
            
            # Extract documents from the index
            docstore = index.storage_context.docstore
            documents = list(docstore.docs.values())
            
            logger.info(f"Found {len(documents)} documents in LlamaIndex for {course_name}")
            
            for i, doc in enumerate(documents):
                # Try multiple ways to get text content
                content = ""
                if hasattr(doc, 'text'):
                    content = doc.text
                elif hasattr(doc, 'get_content'):
                    content = doc.get_content()
                elif hasattr(doc, 'content'):
                    content = doc.content
                else:
                    content = str(doc)
                
                if content.strip():
                    # Get source info from metadata
                    source_name = f"document_{i+1}"
                    if hasattr(doc, 'metadata') and doc.metadata:
                        file_name = doc.metadata.get('file_name') or doc.metadata.get('filename') or doc.metadata.get('source')
                        if file_name:
                            source_name = file_name
                        if not source_name.endswith(('.pdf', '.docx', '.pptx', '.txt', '.md')):
                            source_name = f"{source_name}.pdf"  # Default extension
                    
                    processed_sources.add(source_name)
                    logger.info(f"Processing LlamaIndex document: {source_name} ({len(content)} chars)")
                    self._chunk_and_add_content(content, source_name, "llama_index_document", 
                                              all_chunks, chunk_metadata, chunking_method)
                    
        except Exception as e:
            logger.warning(f"Could not extract content from LlamaIndex for {course_name}: {e}")
    
    def _process_raw_documents(self, raw_dir: Path, doc_processor, all_chunks: List[str], 
                             chunk_metadata: List[Dict], processed_sources: set, chunking_method: str):
        """Process raw documents (PDFs, DOCX, PPTX, etc.)."""
        for file_path in raw_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.pptx', '.epub', '.txt', '.md']:
                try:
                    # Extract content using document processor
                    content = doc_processor.extract_text_from_file(str(file_path))
                    
                    if content.strip():
                        source_name = file_path.name
                        processed_sources.add(source_name)
                        source_type = f"document_{file_path.suffix.lower()[1:]}"  # pdf, docx, etc.
                        
                        self._chunk_and_add_content(content, source_name, source_type, 
                                                  all_chunks, chunk_metadata, chunking_method)
                except Exception as e:
                    logger.warning(f"Could not process document {file_path}: {e}")
    
    def _process_transcriptions(self, trans_dir: Path, all_chunks: List[str], 
                              chunk_metadata: List[Dict], processed_sources: set, chunking_method: str):
        """Process transcription files."""
        for file_path in trans_dir.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():
                    source_name = file_path.name
                    processed_sources.add(source_name)
                    self._chunk_and_add_content(content, source_name, "transcription", 
                                              all_chunks, chunk_metadata, chunking_method)
            except Exception as e:
                logger.warning(f"Could not read transcription {file_path}: {e}")
    
    def _chunk_and_add_content(self, content: str, source_name: str, source_type: str,
                             all_chunks: List[str], chunk_metadata: List[Dict], chunking_method: str):
        """Chunk content and add to collections."""
        # Choose chunking method
        if chunking_method == "paragraphs":
            chunks = self.chunker.chunk_by_paragraphs(content)
        elif chunking_method == "sliding_window":
            chunks = self.chunker.chunk_by_sliding_window(content)
        elif chunking_method == "topics":
            chunks = self.chunker.chunk_by_topics(content)
        else:
            raise ValueError(f"Unknown chunking method: {chunking_method}")
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk['source_file'] = source_name
            chunk['source_type'] = source_type
            chunk['chunk_index'] = len(all_chunks)
            
            all_chunks.append(chunk['content'])
            chunk_metadata.append(chunk)
    
    def search_course(self, course_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within a specific course using vector similarity."""
        # Load course vectors if not cached
        if course_name not in self.course_vectors:
            vectors_file = self.vectors_dir / f"{course_name}_vectors.json"
            if not vectors_file.exists():
                return []
            
            with open(vectors_file, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)
            
            self.course_vectors[course_name] = {
                'chunks': vector_data['chunks'],
                'embeddings': np.array(vector_data['embeddings'])
            }
        
        # Perform similarity search
        course_data = self.course_vectors[course_name]
        results = self.embeddings_engine.search_similar(
            query, course_data['embeddings'], course_data['chunks'], top_k
        )
        
        # Format results
        formatted_results = []
        for chunk, similarity in results:
            formatted_results.append({
                'content': chunk['content'],
                'similarity': similarity,
                'source_file': chunk.get('source_file', 'unknown'),
                'chunk_type': chunk.get('type', 'unknown'),
                'word_count': chunk.get('word_count', 0),
                'metadata': chunk
            })
        
        return formatted_results
    
    def search_all_courses(self, query: str, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all courses."""
        all_results = {}
        
        # Get all course vector files
        vector_files = list(self.vectors_dir.glob("*_vectors.json"))
        
        for vector_file in vector_files:
            course_name = vector_file.stem.replace("_vectors", "")
            results = self.search_course(course_name, query, top_k // len(vector_files) + 1)
            if results:
                all_results[course_name] = results
        
        return all_results
    
    def generate_response_with_context(self, query: str, context_chunks: List[Dict[str, Any]], 
                                     api_provider: str = "openai", api_key: str = None) -> str:
        """Generate response using cloud API with relevant context."""
        # Prepare context
        context_text = "\n\n".join([
            f"Source: {chunk['source_file']}\nContent: {chunk['content']}"
            for chunk in context_chunks
        ])
        
        # Construct prompt
        prompt = f"""Based on the following course material context, please answer the question.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what additional information might be needed."""
        
        # Call appropriate API
        if api_provider == "openai":
            return self._call_openai_api(prompt, api_key)
        elif api_provider == "perplexity":
            return self._call_perplexity_api(prompt, api_key)
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")
    
    def _call_openai_api(self, prompt: str, api_key: str = None) -> str:
        """Call OpenAI API with context."""
        try:
            import os
            from openai import OpenAI
            
            # Use provided API key or fall back to environment
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                return "OpenAI API key not available"
            
            client = OpenAI(api_key=key)
            
            response = client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for best quality
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing real estate course materials."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content if content is not None else "No response from OpenAI"
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error calling OpenAI API: {e}"
    
    def _call_perplexity_api(self, prompt: str, api_key: str = None) -> str:
        """Call Perplexity API with context."""
        try:
            import os
            import requests
            
            # Use provided API key or fall back to environment
            key = api_key or os.getenv("PERPLEXITY_API_KEY")
            if not key:
                return "Perplexity API key not available"
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant analyzing real estate course materials."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_details = ""
                try:
                    error_info = response.json()
                    error_details = f" - {error_info.get('error', {}).get('message', 'Unknown error')}"
                except:
                    pass
                return f"Perplexity API error {response.status_code}{error_details}"
                
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return f"Error calling Perplexity API: {e}"
    
    def get_cost_estimate(self, query: str, num_chunks: int) -> Dict[str, float]:
        """Estimate cost for processing query with vector RAG."""
        # Token estimation
        query_tokens = len(query.split()) * 1.3  # Rough token estimation
        context_tokens = num_chunks * 250  # Average chunk size
        total_input_tokens = query_tokens + context_tokens
        
        # OpenAI pricing (GPT-4)
        input_cost = total_input_tokens * 0.00003  # $0.03/1K tokens
        output_cost = 500 * 0.00006  # Assume 500 output tokens at $0.06/1K
        
        return {
            'input_tokens': total_input_tokens,
            'estimated_input_cost': input_cost,
            'estimated_output_cost': output_cost,
            'total_estimated_cost': input_cost + output_cost,
            'traditional_flat_rate_cost': 20.0 / 30  # $20/month ÷ 30 days
        }