"""
Query Cache System
Caches frequent questions and responses to improve performance and reduce API costs.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)

class QueryCache:
    """Smart caching system for course queries and responses."""
    
    def __init__(self, config=None):
        """Initialize query cache."""
        from config import Config
        self.config = config or Config()
        
        # Create cache directory
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.query_cache_file = self.cache_dir / "query_cache.json"
        self.response_cache_file = self.cache_dir / "response_cache.pickle"
        self.stats_file = self.cache_dir / "cache_stats.json"
        
        # Cache settings
        self.max_cache_size = 1000  # Maximum cached queries
        self.cache_ttl_days = 30    # Cache time-to-live
        self.similarity_threshold = 0.85  # Query similarity threshold
        
        # Load existing cache
        self.query_cache = self._load_query_cache()
        self.response_cache = self._load_response_cache()
        self.stats = self._load_stats()
    
    def _load_query_cache(self) -> Dict[str, Any]:
        """Load query cache metadata."""
        try:
            if self.query_cache_file.exists():
                with open(self.query_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load query cache: {e}")
        
        return {
            "version": "1.0",
            "queries": {},
            "created": datetime.now().isoformat()
        }
    
    def _load_response_cache(self) -> Dict[str, str]:
        """Load response cache data."""
        try:
            if self.response_cache_file.exists():
                with open(self.response_cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load response cache: {e}")
        
        return {}
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load cache statistics."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache stats: {e}")
        
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "cache_saves": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_query_cache(self):
        """Save query cache metadata."""
        try:
            self.query_cache["last_updated"] = datetime.now().isoformat()
            with open(self.query_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.query_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save query cache: {e}")
    
    def _save_response_cache(self):
        """Save response cache data."""
        try:
            with open(self.response_cache_file, 'wb') as f:
                pickle.dump(self.response_cache, f)
        except Exception as e:
            logger.error(f"Failed to save response cache: {e}")
    
    def _save_stats(self):
        """Save cache statistics."""
        try:
            self.stats["last_updated"] = datetime.now().isoformat()
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache stats: {e}")
    
    def _get_query_hash(self, query: str, course_name: str = "") -> str:
        """Generate hash for query + course combination."""
        content = f"{course_name}:{query.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching."""
        # Remove extra whitespace, convert to lowercase
        normalized = " ".join(query.lower().strip().split())
        
        # Remove common question words that don't affect meaning
        stop_words = ["what", "is", "the", "how", "can", "you", "please", "tell", "me", "about"]
        words = normalized.split()
        
        # Keep at least 3 words even if they're stop words
        if len(words) > 3:
            words = [w for w in words if w not in stop_words or len(words) <= 3]
        
        return " ".join(words)
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        # Simple word overlap similarity
        words1 = set(self._normalize_query(query1).split())
        words2 = set(self._normalize_query(query2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _is_cache_valid(self, cached_time: str) -> bool:
        """Check if cached entry is still valid."""
        try:
            cached_dt = datetime.fromisoformat(cached_time)
            expiry_dt = cached_dt + timedelta(days=self.cache_ttl_days)
            return datetime.now() < expiry_dt
        except Exception:
            return False
    
    def find_similar_query(self, query: str, course_name: str = "") -> Optional[Tuple[str, str, float]]:
        """Find similar cached query."""
        normalized_query = self._normalize_query(query)
        best_match = None
        best_similarity = 0.0
        best_hash = None
        
        for query_hash, query_info in self.query_cache["queries"].items():
            if query_info.get("course_name", "") != course_name:
                continue
            
            if not self._is_cache_valid(query_info.get("cached_time", "")):
                continue
            
            cached_query = query_info.get("normalized_query", "")
            similarity = self._calculate_similarity(normalized_query, cached_query)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = query_info.get("original_query", "")
                best_hash = query_hash
        
        if best_match and best_hash:
            return best_hash, best_match, best_similarity
        
        return None
    
    def get_cached_response(self, query: str, course_name: str = "") -> Optional[Dict[str, Any]]:
        """Get cached response for query."""
        self.stats["total_queries"] += 1
        
        # Try exact match first
        query_hash = self._get_query_hash(query, course_name)
        
        if query_hash in self.query_cache["queries"]:
            query_info = self.query_cache["queries"][query_hash]
            
            if self._is_cache_valid(query_info.get("cached_time", "")):
                if query_hash in self.response_cache:
                    self.stats["cache_hits"] += 1
                    self._save_stats()
                    
                    logger.info(f"Cache hit (exact): {query[:50]}...")
                    return {
                        "response": self.response_cache[query_hash],
                        "cached": True,
                        "cache_type": "exact",
                        "similarity": 1.0,
                        "cached_time": query_info["cached_time"]
                    }
        
        # Try similarity match
        similar_match = self.find_similar_query(query, course_name)
        if similar_match:
            similar_hash, similar_query, similarity = similar_match
            
            if similar_hash in self.response_cache:
                self.stats["cache_hits"] += 1
                self._save_stats()
                
                logger.info(f"Cache hit (similar {similarity:.2f}): {query[:50]}...")
                return {
                    "response": self.response_cache[similar_hash],
                    "cached": True,
                    "cache_type": "similar",
                    "similarity": similarity,
                    "original_query": similar_query,
                    "cached_time": self.query_cache["queries"][similar_hash]["cached_time"]
                }
        
        # Cache miss
        self.stats["cache_misses"] += 1
        self._save_stats()
        
        logger.info(f"Cache miss: {query[:50]}...")
        return None
    
    def cache_response(self, query: str, response: str, course_name: str = "",
                      response_time: float = 0.0, model_used: str = "") -> bool:
        """Cache query response."""
        try:
            query_hash = self._get_query_hash(query, course_name)
            current_time = datetime.now().isoformat()
            
            # Store query metadata
            self.query_cache["queries"][query_hash] = {
                "original_query": query,
                "normalized_query": self._normalize_query(query),
                "course_name": course_name,
                "cached_time": current_time,
                "response_time": response_time,
                "model_used": model_used,
                "access_count": 1
            }
            
            # Store response
            self.response_cache[query_hash] = response
            
            # Clean up old entries if cache is too large
            self._cleanup_cache()
            
            # Save caches
            self._save_query_cache()
            self._save_response_cache()
            
            self.stats["cache_saves"] += 1
            self._save_stats()
            
            logger.info(f"Cached response: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
            return False
    
    def _cleanup_cache(self):
        """Clean up old or excess cache entries."""
        queries = self.query_cache["queries"]
        
        # Remove expired entries
        current_time = datetime.now()
        expired_hashes = []
        
        for query_hash, query_info in queries.items():
            if not self._is_cache_valid(query_info.get("cached_time", "")):
                expired_hashes.append(query_hash)
        
        for query_hash in expired_hashes:
            del queries[query_hash]
            if query_hash in self.response_cache:
                del self.response_cache[query_hash]
        
        # If still too large, remove least recently used
        if len(queries) > self.max_cache_size:
            # Sort by access time (oldest first)
            sorted_queries = sorted(
                queries.items(),
                key=lambda x: x[1].get("cached_time", "")
            )
            
            # Remove oldest entries
            entries_to_remove = len(queries) - self.max_cache_size
            for i in range(entries_to_remove):
                query_hash = sorted_queries[i][0]
                del queries[query_hash]
                if query_hash in self.response_cache:
                    del self.response_cache[query_hash]
        
        if expired_hashes or len(queries) > self.max_cache_size:
            logger.info(f"Cleaned up cache: removed {len(expired_hashes)} expired entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats["total_queries"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_queries"]
        
        return {
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": hit_rate,
            "cached_responses": len(self.response_cache),
            "cache_size_mb": self._get_cache_size(),
            "last_updated": self.stats.get("last_updated", "")
        }
    
    def _get_cache_size(self) -> float:
        """Get cache size in MB."""
        try:
            total_size = 0
            for file_path in [self.query_cache_file, self.response_cache_file, self.stats_file]:
                if file_path.exists():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            self.query_cache = {
                "version": "1.0",
                "queries": {},
                "created": datetime.now().isoformat()
            }
            self.response_cache = {}
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_queries": 0,
                "cache_saves": 0,
                "last_updated": datetime.now().isoformat()
            }
            
            # Remove cache files
            for file_path in [self.query_cache_file, self.response_cache_file, self.stats_file]:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_popular_queries(self, course_name: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular cached queries."""
        queries = []
        
        for query_hash, query_info in self.query_cache["queries"].items():
            if course_name and query_info.get("course_name", "") != course_name:
                continue
            
            queries.append({
                "query": query_info.get("original_query", ""),
                "course": query_info.get("course_name", ""),
                "access_count": query_info.get("access_count", 1),
                "cached_time": query_info.get("cached_time", ""),
                "model_used": query_info.get("model_used", "")
            })
        
        # Sort by access count (descending)
        queries.sort(key=lambda x: x["access_count"], reverse=True)
        
        return queries[:limit]