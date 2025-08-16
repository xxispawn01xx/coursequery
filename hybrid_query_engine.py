"""
Hybrid Query Engine - Handles both local and cloud AI models
Optimized for transcription workflow with cloud API querying
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class HybridQueryEngine:
    """Query engine that works with local transcription + cloud APIs."""
    
    def __init__(self):
        """Initialize hybrid query engine."""
        self.config = self._load_config()
        self.available_models = self._detect_available_models()
        self.query_cache = self._init_cache()
        
    def _load_config(self):
        """Load configuration with fallbacks."""
        try:
            from config import Config
            return Config()
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return None
    
    def _detect_available_models(self) -> Dict[str, bool]:
        """Detect what models/services are available."""
        available = {
            "local_embeddings": False,
            "local_llm": False,
            "whisper": False,
            "openai_api": bool(os.getenv("OPENAI_API_KEY")),
            "perplexity_api": bool(os.getenv("PERPLEXITY_API_KEY"))
        }
        
        # Check local models
        try:
            from sentence_transformers import SentenceTransformer
            available["local_embeddings"] = True
        except ImportError:
            logger.info("Local embeddings not available - will use cloud APIs")
        
        # Check for local LLM capability
        try:
            import torch
            from transformers import AutoTokenizer
            if torch.cuda.is_available():
                available["local_llm"] = True
        except ImportError:
            logger.info("Local LLM not available - will use cloud APIs")
        
        # Check for Whisper specifically (local installation required)
        try:
            import whisper
            available["whisper"] = True
            logger.info(" Whisper available for local transcription")
        except ImportError:
            logger.info(" Whisper not installed - install locally with: pip install openai-whisper")
            logger.info(" Once installed locally, Whisper will run on your RTX 3060 for offline transcription")
        
        logger.info(f"Available models: {available}")
        return available
    
    def _init_cache(self):
        """Initialize query cache."""
        try:
            from query_cache import QueryCache
            return QueryCache()
        except Exception as e:
            logger.warning(f"Query cache not available: {e}")
            return None
    
    def process_documents(self, documents: List[Dict], course_name: str) -> Dict[str, Any]:
        """Process documents for a course using best available method."""
        results = {
            "processed_files": 0,
            "transcribed_files": 0,
            "text_files": 0,
            "errors": [],
            "method": "hybrid",
            "timestamp": datetime.now().isoformat()
        }
        
        for doc in documents:
            try:
                file_path = doc.get("file_path", "")
                file_type = doc.get("file_type", "")
                
                if file_type in ["audio", "video"]:
                    if self.available_models["whisper"]:
                        # Use local Whisper for transcription
                        transcription = self._transcribe_locally(file_path)
                        doc["transcribed_content"] = transcription
                        results["transcribed_files"] += 1
                    else:
                        # Use cloud transcription
                        transcription = self._transcribe_cloud(file_path)
                        doc["transcribed_content"] = transcription
                        results["transcribed_files"] += 1
                else:
                    # Text documents - process locally
                    results["text_files"] += 1
                
                results["processed_files"] += 1
                
            except Exception as e:
                error_msg = f"Failed to process {doc.get('file_path', 'unknown')}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def _transcribe_locally(self, file_path: str) -> str:
        """Transcribe using local Whisper model."""
        try:
            # Check if transcription already exists
            from transcription_manager import TranscriptionManager
            tm = TranscriptionManager()
            
            from pathlib import Path
            file_obj = Path(file_path)
            
            if tm.has_transcription(file_obj, "default"):
                return tm.get_transcription(file_obj, "default")
            
            # Load and use local Whisper
            import whisper
            model = whisper.load_model("medium")  # Good balance of speed/accuracy
            result = model.transcribe(file_path)
            
            transcription = result["text"]
            
            # Cache the transcription
            tm.save_transcription(file_obj, "default", transcription, "whisper_local")
            
            logger.info(f"Transcribed locally: {file_path}")
            return transcription
            
        except Exception as e:
            logger.error(f"Local transcription failed: {e}")
            raise
    
    def _transcribe_cloud(self, file_path: str) -> str:
        """Transcribe using cloud API (OpenAI Whisper)."""
        try:
            if not self.available_models["openai_api"]:
                raise ValueError("OpenAI API key not available for cloud transcription")
            
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            logger.info(f"Transcribed via cloud: {file_path}")
            return transcript
            
        except Exception as e:
            logger.error(f"Cloud transcription failed: {e}")
            raise
    
    def query(self, query: str, course_name: str = "") -> Dict[str, Any]:
        """Process query using best available method."""
        start_time = datetime.now()
        
        # Check cache first
        if self.query_cache:
            cached_result = self.query_cache.get_cached_response(query, course_name)
            if cached_result:
                return {
                    "response": cached_result["response"],
                    "cached": True,
                    "cache_type": cached_result.get("cache_type", "unknown"),
                    "similarity": cached_result.get("similarity", 1.0),
                    "response_time": 0.01,  # Near instant for cached
                    "method": "cache"
                }
        
        # Determine best query method
        if self.available_models["openai_api"] or self.available_models["perplexity_api"]:
            result = self._query_cloud_api(query, course_name)
        elif self.available_models["local_llm"]:
            result = self._query_local(query, course_name)
        else:
            result = {
                "response": " No AI models available. Please configure OpenAI or Perplexity API keys for querying.",
                "cached": False,
                "method": "error",
                "response_time": 0.0
            }
        
        # Cache successful responses
        if self.query_cache and result.get("response") and not result.get("cached"):
            response_time = (datetime.now() - start_time).total_seconds()
            self.query_cache.cache_response(
                query, result["response"], course_name, 
                response_time, result.get("method", "unknown")
            )
        
        return result
    
    def _query_cloud_api(self, query: str, course_name: str) -> Dict[str, Any]:
        """Query using cloud APIs (OpenAI or Perplexity)."""
        try:
            # Prefer OpenAI for cost-effectiveness
            if self.available_models["openai_api"]:
                return self._query_openai(query, course_name)
            elif self.available_models["perplexity_api"]:
                return self._query_perplexity(query, course_name)
            else:
                raise ValueError("No cloud API keys available")
                
        except Exception as e:
            logger.error(f"Cloud API query failed: {e}")
            return {
                "response": f" Cloud API error: {e}",
                "cached": False,
                "method": "cloud_error",
                "response_time": 0.0
            }
    
    def _query_openai(self, query: str, course_name: str) -> Dict[str, Any]:
        """Query using OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Build context from course materials
            context = self._get_course_context(course_name, query)
            
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant analyzing course materials for {course_name}. "
                              f"Use the provided context to answer questions accurately."
                },
                {
                    "role": "user",
                    "content": f"Context from course materials:\n{context}\n\nQuestion: {query}"
                }
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Latest model
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                "response": response.choices[0].message.content,
                "cached": False,
                "method": "openai_api",
                "response_time": 2.0,  # Approximate
                "context_used": bool(context)
            }
            
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            raise
    
    def _query_perplexity(self, query: str, course_name: str) -> Dict[str, Any]:
        """Query using Perplexity API."""
        try:
            import requests
            
            api_key = os.getenv("PERPLEXITY_API_KEY")
            context = self._get_course_context(course_name, query)
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are analyzing course materials for {course_name}. "
                                  f"Combine the course context with current information."
                    },
                    {
                        "role": "user",
                        "content": f"Course context:\n{context}\n\nQuestion: {query}"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
                "search_domain_filter": ["perplexity.ai"],
                "return_related_questions": False
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result["choices"][0]["message"]["content"],
                    "cached": False,
                    "method": "perplexity_api",
                    "response_time": 3.0,  # Approximate
                    "context_used": bool(context)
                }
            else:
                raise ValueError(f"Perplexity API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Perplexity query failed: {e}")
            raise
    
    def _query_local(self, query: str, course_name: str) -> Dict[str, Any]:
        """Query using local models."""
        try:
            # Get course context using local embeddings
            context = self._get_course_context(course_name, query)
            
            # Build response with local processing
            response = f"Local query processed for course '{course_name}'. "
            if context:
                response += f"Found relevant content from course materials. "
            response += "For detailed AI analysis, please configure OpenAI or Perplexity API keys."
            
            return {
                "response": response,
                "cached": False,
                "method": "local_basic",
                "response_time": 0.1
            }
        except Exception as e:
            logger.error(f"Local query failed: {e}")
            return {
                "response": f"Local processing available. Add API keys for AI analysis.",
                "cached": False,
                "method": "local_ready",
                "response_time": 0.0
            }
    
    def _query_with_local_embeddings_only(self, query: str, course_name: str) -> Dict[str, Any]:
        """Query with local embeddings but no local LLM."""
        try:
            context = self._get_course_context(course_name, query)
            response = f"Found course content for '{course_name}'. Add API keys for AI analysis."
            
            return {
                "response": response,
                "cached": False,
                "method": "embeddings_only",
                "response_time": 0.1
            }
        except Exception as e:
            return {
                "response": f"Embeddings processing available. Add API keys for detailed analysis.",
                "cached": False,
                "method": "embeddings_ready",
                "response_time": 0.0
            }
            
        except Exception as e:
            logger.error(f"Local query failed: {e}")
            return {
                "response": " Local models not available. Please configure cloud API keys.",
                "cached": False,
                "method": "local_error",
                "response_time": 0.0
            }
    
    def _get_course_context(self, course_name: str, query: str) -> str:
        """Get relevant context from course materials."""
        try:
            # Try to get context from course indexer
            from course_indexer import CourseIndexer
            indexer = CourseIndexer()
            
            # This would normally retrieve relevant chunks
            # For now, return a basic context message
            return f"Course: {course_name}\nQuery: {query}\n" \
                   f"Note: Full context retrieval requires local embeddings or cloud integration."
        
        except Exception as e:
            logger.warning(f"Could not retrieve course context: {e}")
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "available_models": self.available_models,
            "recommended_workflow": self._get_workflow_recommendation(),
            "cache_stats": self.query_cache.get_cache_stats() if self.query_cache else None
        }
    
    def _get_workflow_recommendation(self) -> str:
        """Get workflow recommendation based on available models."""
        if self.available_models["whisper"] and (
            self.available_models["openai_api"] or self.available_models["perplexity_api"]
        ):
            return "Optimal: Local Whisper transcription + Cloud API querying"
        elif self.available_models["openai_api"] or self.available_models["perplexity_api"]:
            return "Good: Cloud transcription + Cloud API querying"
        elif self.available_models["local_llm"]:
            return "Basic: Local-only processing (consider adding API keys)"
        else:
            return "Limited: Configure API keys for full functionality"