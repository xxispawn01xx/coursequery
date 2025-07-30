"""
Course Rename Handler
Manages course folder renaming and updates associated indexes
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class CourseRenameHandler:
    """Handles course folder renaming and index updates."""
    
    def __init__(self, master_directory: str, indexed_directory: str):
        self.master_directory = Path(master_directory)
        self.indexed_directory = Path(indexed_directory)
    
    def detect_renamed_courses(self) -> List[Dict[str, str]]:
        """
        Detect courses that may have been renamed by comparing
        archived_courses with existing indexes.
        """
        renamed_courses = []
        
        # Get current course directories
        current_courses = set()
        if self.master_directory.exists():
            for item in self.master_directory.iterdir():
                if item.is_dir():
                    current_courses.add(item.name)
        
        # Get indexed course names
        indexed_courses = set()
        if self.indexed_directory.exists():
            for item in self.indexed_directory.iterdir():
                if item.is_dir():
                    indexed_courses.add(item.name)
        
        # Find courses in index but not in current directories
        orphaned_indexes = indexed_courses - current_courses
        new_courses = current_courses - indexed_courses
        
        logger.info(f"Found {len(orphaned_indexes)} orphaned indexes")
        logger.info(f"Found {len(new_courses)} new course directories")
        
        # Try to match orphaned indexes with new courses by similarity
        for orphaned in orphaned_indexes:
            best_match = self._find_best_match(orphaned, new_courses)
            if best_match:
                renamed_courses.append({
                    'old_name': orphaned,
                    'new_name': best_match,
                    'confidence': self._calculate_similarity(orphaned, best_match)
                })
                new_courses.remove(best_match)  # Remove from potential matches
        
        return renamed_courses
    
    def _find_best_match(self, orphaned_name: str, candidates: set) -> Optional[str]:
        """Find the best matching course name for a renamed course."""
        best_match = None
        best_score = 0.5  # Minimum similarity threshold
        
        for candidate in candidates:
            similarity = self._calculate_similarity(orphaned_name, candidate)
            if similarity > best_score:
                best_score = similarity
                best_match = candidate
        
        return best_match
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two course names."""
        # Simple similarity based on common words and characters
        name1_words = set(name1.lower().replace('-', ' ').replace('_', ' ').split())
        name2_words = set(name2.lower().replace('-', ' ').replace('_', ' ').split())
        
        if not name1_words or not name2_words:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(name1_words.intersection(name2_words))
        union = len(name1_words.union(name2_words))
        
        return intersection / union if union > 0 else 0.0
    
    def rename_course_index(self, old_name: str, new_name: str) -> bool:
        """
        Rename a course index directory and update metadata.
        """
        try:
            old_index_path = self.indexed_directory / old_name
            new_index_path = self.indexed_directory / new_name
            
            if not old_index_path.exists():
                logger.warning(f"Old index path does not exist: {old_index_path}")
                return False
            
            if new_index_path.exists():
                logger.warning(f"New index path already exists: {new_index_path}")
                return False
            
            # Rename the index directory
            shutil.move(str(old_index_path), str(new_index_path))
            logger.info(f"Renamed index directory: {old_name} → {new_name}")
            
            # Update metadata files if they exist
            self._update_metadata_files(new_index_path, old_name, new_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error renaming course index: {e}")
            return False
    
    def _update_metadata_files(self, index_path: Path, old_name: str, new_name: str):
        """Update metadata files with new course name."""
        try:
            # Update course metadata JSON if it exists
            metadata_file = index_path / "course_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Update course name references
                if 'course_name' in metadata:
                    metadata['course_name'] = new_name
                if 'original_path' in metadata:
                    metadata['original_path'] = metadata['original_path'].replace(old_name, new_name)
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Updated metadata file: {metadata_file}")
            
            # Update any other config files that might contain the course name
            for config_file in index_path.glob("*.json"):
                if config_file.name != "course_metadata.json":
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Replace old course name with new one
                        updated_content = content.replace(old_name, new_name)
                        
                        if updated_content != content:
                            with open(config_file, 'w', encoding='utf-8') as f:
                                f.write(updated_content)
                            logger.info(f"Updated config file: {config_file}")
                            
                    except Exception as e:
                        logger.warning(f"Could not update config file {config_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error updating metadata files: {e}")
    
    def get_rename_suggestions(self) -> List[Dict[str, str]]:
        """Get suggestions for course renames that need manual confirmation."""
        return self.detect_renamed_courses()
    
    def apply_rename_suggestion(self, old_name: str, new_name: str) -> bool:
        """Apply a rename suggestion after user confirmation."""
        return self.rename_course_index(old_name, new_name)
    
    def cleanup_orphaned_indexes(self) -> List[str]:
        """
        Clean up orphaned indexes that have no corresponding course directory.
        Returns list of cleaned up indexes.
        """
        cleaned_up = []
        
        # Get current course directories
        current_courses = set()
        if self.master_directory.exists():
            for item in self.master_directory.iterdir():
                if item.is_dir():
                    current_courses.add(item.name)
        
        # Find orphaned indexes
        if self.indexed_directory.exists():
            for item in self.indexed_directory.iterdir():
                if item.is_dir() and item.name not in current_courses:
                    try:
                        shutil.rmtree(item)
                        cleaned_up.append(item.name)
                        logger.info(f"Cleaned up orphaned index: {item.name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {item.name}: {e}")
        
        return cleaned_up


def create_rename_handler(master_directory: str, indexed_directory: str) -> CourseRenameHandler:
    """Factory function to create a CourseRenameHandler."""
    return CourseRenameHandler(master_directory, indexed_directory)