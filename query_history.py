"""
Query History Manager
Tracks user queries and responses for each course to show recent activity.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class QueryHistory:
    """Manages query history for each course."""
    
    def __init__(self):
        self.history_dir = Path("cache/query_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_per_course = 20  # Keep last 20 queries per course
        
    def add_query(self, course: str, query: str, response: str, provider: str, 
                  cached: bool = False, metadata: Dict = None) -> bool:
        """Add a query to the history."""
        try:
            history_file = self.history_dir / f"{course}_history.json"
            
            # Load existing history
            history = self._load_history(course)
            
            # Create new entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response[:500] + "..." if len(response) > 500 else response,  # Truncate long responses
                'provider': provider,
                'cached': cached,
                'metadata': metadata or {}
            }
            
            # Add to beginning of list (most recent first)
            history.insert(0, entry)
            
            # Keep only max_history_per_course entries
            history = history[:self.max_history_per_course]
            
            # Save updated history
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Added query to history for course: {course}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding query to history: {e}")
            return False
    
    def get_course_history(self, course: str, limit: int = 10) -> List[Dict]:
        """Get query history for a specific course."""
        try:
            history = self._load_history(course)
            return history[:limit]
            
        except Exception as e:
            logger.error(f"Error getting course history: {e}")
            return []
    
    def get_last_query(self, course: str) -> Optional[Dict]:
        """Get the most recent query for a course."""
        try:
            history = self._load_history(course)
            return history[0] if history else None
            
        except Exception as e:
            logger.error(f"Error getting last query: {e}")
            return None
    
    def get_all_courses_with_history(self) -> List[str]:
        """Get list of courses that have query history."""
        try:
            history_files = list(self.history_dir.glob("*_history.json"))
            courses = [f.stem.replace("_history", "") for f in history_files]
            return sorted(courses)
            
        except Exception as e:
            logger.error(f"Error getting courses with history: {e}")
            return []
    
    def search_history(self, course: str, search_term: str, limit: int = 5) -> List[Dict]:
        """Search query history for a specific term."""
        try:
            history = self._load_history(course)
            
            # Search in queries and responses
            search_term_lower = search_term.lower()
            matches = []
            
            for entry in history:
                if (search_term_lower in entry['query'].lower() or 
                    search_term_lower in entry['response'].lower()):
                    matches.append(entry)
                    
                if len(matches) >= limit:
                    break
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return []
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Get statistics about query history."""
        try:
            stats = {
                'total_courses': 0,
                'total_queries': 0,
                'courses_breakdown': {},
                'provider_usage': {},
                'cache_hit_rate': 0
            }
            
            courses = self.get_all_courses_with_history()
            stats['total_courses'] = len(courses)
            
            total_cached = 0
            total_queries = 0
            
            for course in courses:
                history = self._load_history(course)
                course_query_count = len(history)
                stats['total_queries'] += course_query_count
                stats['courses_breakdown'][course] = course_query_count
                
                # Count provider usage and cache hits
                for entry in history:
                    provider = entry.get('provider', 'unknown')
                    stats['provider_usage'][provider] = stats['provider_usage'].get(provider, 0) + 1
                    
                    if entry.get('cached', False):
                        total_cached += 1
                    
                    total_queries += 1
            
            # Calculate cache hit rate
            if total_queries > 0:
                stats['cache_hit_rate'] = (total_cached / total_queries) * 100
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting history stats: {e}")
            return {}
    
    def clear_course_history(self, course: str) -> bool:
        """Clear history for a specific course."""
        try:
            history_file = self.history_dir / f"{course}_history.json"
            if history_file.exists():
                history_file.unlink()
                logger.info(f"Cleared history for course: {course}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error clearing course history: {e}")
            return False
    
    def clear_all_history(self) -> int:
        """Clear all query history."""
        try:
            history_files = list(self.history_dir.glob("*_history.json"))
            cleared_count = 0
            
            for history_file in history_files:
                history_file.unlink()
                cleared_count += 1
            
            logger.info(f"Cleared history for {cleared_count} courses")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing all history: {e}")
            return 0
    
    def export_history(self, course: str = None) -> Dict:
        """Export history for backup or analysis."""
        try:
            if course:
                # Export specific course
                history = self._load_history(course)
                return {course: history}
            else:
                # Export all courses
                all_history = {}
                courses = self.get_all_courses_with_history()
                
                for course_name in courses:
                    all_history[course_name] = self._load_history(course_name)
                
                return all_history
                
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return {}
    
    def _load_history(self, course: str) -> List[Dict]:
        """Load history for a specific course."""
        try:
            history_file = self.history_dir / f"{course}_history.json"
            
            if not history_file.exists():
                return []
            
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading history for course {course}: {e}")
            return []
    
    def get_recent_queries_across_courses(self, limit: int = 10) -> List[Dict]:
        """Get most recent queries across all courses."""
        try:
            all_entries = []
            courses = self.get_all_courses_with_history()
            
            for course in courses:
                history = self._load_history(course)
                for entry in history:
                    entry['course'] = course
                    all_entries.append(entry)
            
            # Sort by timestamp (most recent first)
            all_entries.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return all_entries[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent queries: {e}")
            return []