"""
Response Cache Manager
Caches RAG responses to avoid duplicate API calls and improve performance.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class ResponseCache:
    """Manages caching of RAG query responses to reduce API costs."""
    
    def __init__(self):
        self.cache_dir = Path("cache/responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = 24  # Cache responses for 24 hours
        
    def _generate_cache_key(self, query: str, course: str, provider: str, context_chunks: list = None) -> str:
        """Generate unique cache key for query parameters."""
        # Create hash from query, course, provider, and context
        cache_input = {
            'query': query.lower().strip(),
            'course': course,
            'provider': provider,
            'context_hash': hashlib.md5(str(context_chunks).encode()).hexdigest() if context_chunks else None
        }
        
        cache_string = json.dumps(cache_input, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_response(self, query: str, course: str, provider: str, context_chunks: list = None) -> Optional[Dict]:
        """Retrieve cached response if available and not expired."""
        try:
            cache_key = self._generate_cache_key(query, course, provider, context_chunks)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if not cache_file.exists():
                return None
            
            # Load cached data
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.max_age_hours):
                # Remove expired cache
                cache_file.unlink()
                logger.info(f"Expired cache removed: {cache_key}")
                return None
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_data
            
        except Exception as e:
            logger.error(f"Error retrieving cached response: {e}")
            return None
    
    def cache_response(self, query: str, course: str, provider: str, response: str, 
                      context_chunks: list = None, metadata: Dict = None) -> bool:
        """Cache a response for future use."""
        try:
            cache_key = self._generate_cache_key(query, course, provider, context_chunks)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            cache_data = {
                'query': query,
                'course': course,
                'provider': provider,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'context_chunks_count': len(context_chunks) if context_chunks else 0
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Response cached: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            
            total_files = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Count by provider
            provider_counts = {}
            expired_count = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    provider = data.get('provider', 'unknown')
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1
                    
                    # Check if expired
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - cached_time > timedelta(hours=self.max_age_hours):
                        expired_count += 1
                        
                except Exception:
                    continue
            
            return {
                'total_cached_responses': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'provider_breakdown': provider_counts,
                'expired_responses': expired_count,
                'cache_directory': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            cleared_count = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - cached_time > timedelta(hours=self.max_age_hours):
                        cache_file.unlink()
                        cleared_count += 1
                        
                except Exception:
                    # If we can't read the file, remove it
                    cache_file.unlink()
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} expired cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0
    
    def clear_all_cache(self) -> int:
        """Clear all cached responses."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            cleared_count = 0
            
            for cache_file in cache_files:
                cache_file.unlink()
                cleared_count += 1
            
            logger.info(f"Cleared all {cleared_count} cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return 0
    
    def find_similar_queries(self, query: str, limit: int = 5) -> list:
        """Find similar cached queries for suggestion."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            similar_queries = []
            
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    cached_query = data.get('query', '').lower()
                    cached_words = set(cached_query.split())
                    
                    # Calculate similarity (Jaccard similarity)
                    intersection = len(query_words.intersection(cached_words))
                    union = len(query_words.union(cached_words))
                    
                    if union > 0:
                        similarity = intersection / union
                        if similarity > 0.3:  # 30% similarity threshold
                            similar_queries.append({
                                'query': data.get('query'),
                                'similarity': similarity,
                                'provider': data.get('provider'),
                                'timestamp': data.get('timestamp')
                            })
                            
                except Exception:
                    continue
            
            # Sort by similarity and return top results
            similar_queries.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_queries[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []