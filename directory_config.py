"""
Centralized Directory Configuration Manager
Single source of truth for all course directory paths throughout the system
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# =====================================
# MASTER ROOT DIRECTORY CONFIGURATION
# =====================================
# Single place to change the root coursequery directory
ROOT_COURSEQUERY_DIRECTORY = r"H:\coursequery"

class DirectoryConfigManager:
    """Manages centralized directory configuration for the entire system."""
    
    def __init__(self):
        """Initialize with master directory configuration."""
        # =====================================
        # MASTER COURSE DIRECTORY CONFIGURATION
        # =====================================
        # First try to find coursequery directory dynamically
        self.MASTER_COURSE_DIRECTORY = self.find_coursequery_directory()
        
        # Configuration file for persistence
        self.config_file = Path(__file__).parent / "directory_config.json"
        
        # Load any saved configuration
        self.load_configuration()
        
        # Set up all derived paths
        self.setup_all_paths()
    
    def find_coursequery_directory(self):
        """Find archived_courses directory relative to where streamlit is launched."""
        
        # Start from current working directory (where streamlit was launched)
        cwd = Path.cwd()
        
        # Check if archived_courses exists in current directory
        archived_courses_path = cwd / "archived_courses"
        if archived_courses_path.exists():
            print(f"📁 Found archived_courses in current directory: {archived_courses_path}")
            return str(archived_courses_path)
        
        # Check if we're inside coursequery directory already
        if cwd.name == "coursequery" and archived_courses_path.exists():
            print(f"📁 Running from coursequery directory: {archived_courses_path}")
            return str(archived_courses_path)
        
        # Check parent directory for archived_courses
        parent_archived = cwd.parent / "archived_courses"
        if parent_archived.exists():
            print(f"📁 Found archived_courses in parent directory: {parent_archived}")
            return str(parent_archived)
        
        # Default: create archived_courses in current directory if H:\ not accessible
        if not os.path.exists(ROOT_COURSEQUERY_DIRECTORY):
            print(f"📁 Root directory not accessible ({ROOT_COURSEQUERY_DIRECTORY}), using local: {archived_courses_path}")
            return str(archived_courses_path)
        
        # Fallback to configured root location
        print(f"📁 Using configured root location: {ROOT_COURSEQUERY_DIRECTORY}")
        return os.path.join(ROOT_COURSEQUERY_DIRECTORY, "archived_courses")
    
    def load_configuration(self):
        """Load saved directory configuration if available."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                # Override master directory if saved configuration exists
                if 'master_directory' in config_data:
                    self.MASTER_COURSE_DIRECTORY = config_data['master_directory']
                    print(f"📁 Loaded saved directory: {self.MASTER_COURSE_DIRECTORY}")
        except Exception as e:
            print(f"⚠️ Could not load directory config: {e}")
    
    def save_configuration(self):
        """Save current directory configuration."""
        try:
            config_data = {
                'master_directory': self.MASTER_COURSE_DIRECTORY,
                'last_updated': str(Path.now() if hasattr(Path, 'now') else 'unknown')
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved directory configuration: {self.MASTER_COURSE_DIRECTORY}")
            return True
        except Exception as e:
            print(f"⚠️ Could not save directory config: {e}")
            return False
    
    def setup_all_paths(self):
        """Setup all system paths based on master directory."""
        # Master course directory
        self.master_path = Path(self.MASTER_COURSE_DIRECTORY)
        
        # Check if master directory exists
        if self.master_path.exists():
            self.raw_docs_dir = self.master_path
            self.using_master = True
            print(f"📁 Using MASTER course directory: {self.raw_docs_dir}")
        else:
            # Fallback to local directory
            self.raw_docs_dir = Path(__file__).parent / "archived_courses"
            self.using_master = False
            print(f"📁 Using FALLBACK directory: {self.raw_docs_dir}")
            print(f"⚠️ Master directory not accessible: {self.MASTER_COURSE_DIRECTORY}")
        
        # Derived paths (always local for processing)
        base_dir = Path(__file__).parent
        self.indexed_courses_dir = base_dir / "indexed_courses"
        self.book_embeddings_dir = base_dir / "book_embeddings"
        self.models_dir = base_dir / "models"
        self.cache_dir = base_dir / "cache"
        self.temp_dir = base_dir / "temp"
        
        # Create processing directories
        for directory in [self.indexed_courses_dir, self.book_embeddings_dir, 
                         self.models_dir, self.cache_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
        
        # Create raw docs directory if using fallback
        if not self.using_master:
            self.raw_docs_dir.mkdir(exist_ok=True)
    
    def update_master_directory(self, new_path: str) -> bool:
        """Update the master directory path and cascade changes."""
        try:
            new_path_obj = Path(new_path)
            
            # Validate the new path
            if not new_path_obj.exists():
                print(f"❌ Directory does not exist: {new_path}")
                return False
            
            if not new_path_obj.is_dir():
                print(f"❌ Path is not a directory: {new_path}")
                return False
            
            # Update master directory
            old_path = self.MASTER_COURSE_DIRECTORY
            self.MASTER_COURSE_DIRECTORY = str(new_path_obj)
            
            # Reconfigure all paths
            self.setup_all_paths()
            
            # Save configuration
            self.save_configuration()
            
            print(f"✅ Updated master directory:")
            print(f"   From: {old_path}")
            print(f"   To:   {self.MASTER_COURSE_DIRECTORY}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating master directory: {e}")
            return False
    
    def get_course_path(self, course_name: str) -> Path:
        """Get the full path to a specific course."""
        return self.raw_docs_dir / course_name
    
    def get_indexed_course_path(self, course_name: str) -> Path:
        """Get the path to indexed course data."""
        return self.indexed_courses_dir / course_name
    
    def get_book_embedding_path(self, book_id: str) -> Path:
        """Get the path to book embedding data."""
        return self.book_embeddings_dir / book_id
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current directory configuration."""
        return {
            'master_directory': self.MASTER_COURSE_DIRECTORY,
            'using_master': self.using_master,
            'actual_raw_docs': str(self.raw_docs_dir),
            'indexed_courses': str(self.indexed_courses_dir),
            'book_embeddings': str(self.book_embeddings_dir),
            'models': str(self.models_dir),
            'cache': str(self.cache_dir),
            'temp': str(self.temp_dir),
            'config_file': str(self.config_file)
        }
    
    def list_available_courses(self) -> list:
        """List all available courses in the master directory."""
        courses = []
        
        try:
            if self.raw_docs_dir.exists():
                for item in self.raw_docs_dir.iterdir():
                    if item.is_dir():
                        # Count files in the course directory
                        file_count = len([f for f in item.rglob('*') if f.is_file()])
                        courses.append({
                            'name': item.name,
                            'path': str(item),
                            'file_count': file_count
                        })
        except Exception as e:
            print(f"❌ Error listing courses: {e}")
        
        return sorted(courses, key=lambda x: x['name'])
    
    def validate_directory_access(self) -> Dict[str, bool]:
        """Validate access to all configured directories."""
        validation = {}
        
        directories = {
            'master_directory': self.master_path,
            'raw_docs': self.raw_docs_dir,
            'indexed_courses': self.indexed_courses_dir,
            'book_embeddings': self.book_embeddings_dir,
            'models': self.models_dir,
            'cache': self.cache_dir,
            'temp': self.temp_dir
        }
        
        for name, path in directories.items():
            try:
                validation[name] = {
                    'exists': path.exists(),
                    'is_dir': path.is_dir() if path.exists() else False,
                    'readable': os.access(path, os.R_OK) if path.exists() else False,
                    'writable': os.access(path, os.W_OK) if path.exists() else False,
                    'path': str(path)
                }
            except Exception as e:
                validation[name] = {
                    'exists': False,
                    'is_dir': False,
                    'readable': False,
                    'writable': False,
                    'path': str(path),
                    'error': str(e)
                }
        
        return validation

# Global instance for system-wide use
directory_config = DirectoryConfigManager()

def get_master_directory() -> str:
    """Get the master course directory path."""
    return directory_config.MASTER_COURSE_DIRECTORY

def update_master_directory(new_path: str) -> bool:
    """Update the master directory path system-wide."""
    return directory_config.update_master_directory(new_path)

def get_course_path(course_name: str) -> Path:
    """Get the full path to a specific course."""
    return directory_config.get_course_path(course_name)

def get_directory_config() -> DirectoryConfigManager:
    """Get the global directory configuration manager."""
    return directory_config