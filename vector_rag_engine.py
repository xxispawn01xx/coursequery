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
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            self.load_model()
        
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
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.embeddings_cache[text_hash] = embedding
        
        # Combine cached and new embeddings in correct order
        all_embeddings = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        
        # Add cached embeddings
        for original_idx, embedding in cached_embeddings:
            all_embeddings[original_idx] = embedding
        
        # Add new embeddings
        if uncached_texts:
            for i, original_idx in enumerate(uncached_indices):
                all_embeddings[original_idx] = new_embeddings[i]
        
        return all_embeddings
    
    def search_similar(self, query_text: str, chunk_embeddings: np.ndarray, 
                      chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for most similar chunks to query."""
        if self.model is None:
            self.load_model()
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        
        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
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
                                     api_provider: str = "openai") -> str:
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
            return self._call_openai_api(prompt)
        elif api_provider == "perplexity":
            return self._call_perplexity_api(prompt)
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with context."""
        try:
            import os
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for best quality
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing real estate course materials."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error calling OpenAI API: {e}"
    
    def _call_perplexity_api(self, prompt: str) -> str:
        """Call Perplexity API with context."""
        try:
            import os
            import requests
            
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key:
                return "Perplexity API key not found"
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant analyzing real estate course materials."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Perplexity API error: {response.status_code}"
                
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