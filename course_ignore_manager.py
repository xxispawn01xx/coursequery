"""
Course Ignore Manager - Handle ignored/hidden courses functionality
"""
import json
from pathlib import Path
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class CourseIgnoreManager:
    """Manages ignored courses list with persistent storage."""
    
    def __init__(self, config_dir: str = "."):
        self.config_dir = Path(config_dir)
        self.ignore_file = self.config_dir / "ignored_courses.json"
        self._ignored_courses: Set[str] = set()
        self.load_ignored_courses()
    
    def load_ignored_courses(self) -> None:
        """Load ignored courses from persistent storage."""
        try:
            if self.ignore_file.exists():
                with open(self.ignore_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._ignored_courses = set(data.get('ignored_courses', []))
                    logger.info(f"Loaded {len(self._ignored_courses)} ignored courses")
            else:
                self._ignored_courses = set()
                logger.info("No ignored courses file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading ignored courses: {e}")
            self._ignored_courses = set()
    
    def save_ignored_courses(self) -> None:
        """Save ignored courses to persistent storage."""
        try:
            self.config_dir.mkdir(exist_ok=True)
            data = {
                'ignored_courses': list(self._ignored_courses),
                'last_updated': str(Path.cwd())  # Store current timestamp-like info
            }
            with open(self.ignore_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self._ignored_courses)} ignored courses")
        except Exception as e:
            logger.error(f"Error saving ignored courses: {e}")
    
    def ignore_course(self, course_name: str) -> bool:
        """Add a course to the ignore list."""
        if course_name not in self._ignored_courses:
            self._ignored_courses.add(course_name)
            self.save_ignored_courses()
            logger.info(f"Added '{course_name}' to ignore list")
            return True
        return False
    
    def unignore_course(self, course_name: str) -> bool:
        """Remove a course from the ignore list."""
        if course_name in self._ignored_courses:
            self._ignored_courses.remove(course_name)
            self.save_ignored_courses()
            logger.info(f"Removed '{course_name}' from ignore list")
            return True
        return False
    
    def is_ignored(self, course_name: str) -> bool:
        """Check if a course is ignored."""
        return course_name in self._ignored_courses
    
    def get_ignored_courses(self) -> List[str]:
        """Get list of all ignored courses."""
        return sorted(list(self._ignored_courses))
    
    def filter_courses(self, courses: List[dict]) -> List[dict]:
        """Filter out ignored courses from a course list."""
        return [course for course in courses if not self.is_ignored(course['name'])]
    
    def get_stats(self) -> dict:
        """Get ignore manager statistics."""
        return {
            'total_ignored': len(self._ignored_courses),
            'ignored_courses': self.get_ignored_courses(),
            'config_file': str(self.ignore_file),
            'config_exists': self.ignore_file.exists()
        }