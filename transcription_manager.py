"""
Transcription Management System
Handles transcription storage, retrieval, and API model integration.
Preserves original folder structure while avoiding re-processing media files.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class TranscriptionManager:
    """Manages transcription storage and retrieval to avoid re-processing media files."""
    
    def __init__(self, config=None):
        """Initialize transcription manager."""
        from config import Config
        self.config = config or Config()
        
        # Create transcriptions directory structure
        self.transcriptions_dir = Path("./transcriptions")
        self.transcriptions_dir.mkdir(exist_ok=True)
        
        # Metadata file for tracking transcriptions
        self.metadata_file = self.transcriptions_dir / "transcription_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load transcription metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load transcription metadata: {e}")
        
        return {
            "version": "1.0",
            "transcriptions": {},
            "created": datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save transcription metadata to file."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save transcription metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to detect changes."""
        try:
            # Use file size + modification time for quick hash
            stat = file_path.stat()
            content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not generate hash for {file_path}: {e}")
            return str(file_path)
    
    def get_transcription_path(self, media_file: Path, course_name: str) -> Path:
        """Get the path where transcription should be stored."""
        # Preserve folder structure under transcriptions directory
        relative_path = media_file.relative_to(media_file.parent.parent)
        
        # Create course-specific transcription directory
        course_dir = self.transcriptions_dir / course_name
        course_dir.mkdir(exist_ok=True)
        
        # Change extension to .txt
        transcription_file = course_dir / f"{media_file.stem}.txt"
        return transcription_file
    
    def has_transcription(self, media_file: Path, course_name: str) -> bool:
        """Check if transcription already exists and is up to date."""
        file_hash = self._get_file_hash(media_file)
        file_key = str(media_file)
        
        # Check metadata
        if file_key in self.metadata["transcriptions"]:
            stored_info = self.metadata["transcriptions"][file_key]
            
            # Check if file hasn't changed
            if stored_info.get("file_hash") == file_hash:
                transcription_path = Path(stored_info.get("transcription_path", ""))
                
                # Check if transcription file still exists
                if transcription_path.exists():
                    logger.info(f"Found existing transcription: {media_file.name}")
                    return True
        
        return False
    
    def get_transcription(self, media_file: Path, course_name: str) -> Optional[str]:
        """Get existing transcription content."""
        file_key = str(media_file)
        
        if file_key in self.metadata["transcriptions"]:
            stored_info = self.metadata["transcriptions"][file_key]
            transcription_path = Path(stored_info.get("transcription_path", ""))
            
            try:
                if transcription_path.exists():
                    with open(transcription_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.info(f"Loaded transcription: {media_file.name} ({len(content)} chars)")
                    return content
            except Exception as e:
                logger.error(f"Failed to read transcription {transcription_path}: {e}")
        
        return None
    
    def save_transcription(self, media_file: Path, course_name: str, transcription: str, 
                          method: str = "whisper") -> bool:
        """Save transcription to file and update metadata."""
        try:
            transcription_path = self.get_transcription_path(media_file, course_name)
            transcription_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save transcription content
            with open(transcription_path, 'w', encoding='utf-8') as f:
                # Add header with metadata
                header = f"# Transcription: {media_file.name}\n"
                header += f"# Course: {course_name}\n"
                header += f"# Method: {method}\n"
                header += f"# Generated: {datetime.now().isoformat()}\n"
                header += f"# Original file: {media_file}\n\n"
                
                f.write(header + transcription)
            
            # Update metadata
            file_key = str(media_file)
            self.metadata["transcriptions"][file_key] = {
                "file_hash": self._get_file_hash(media_file),
                "transcription_path": str(transcription_path),
                "course_name": course_name,
                "method": method,
                "created": datetime.now().isoformat(),
                "character_count": len(transcription),
                "original_file": str(media_file)
            }
            
            self._save_metadata()
            
            logger.info(f"Saved transcription: {media_file.name} -> {transcription_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transcription for {media_file}: {e}")
            return False
    
    def get_all_transcriptions(self, course_name: str) -> List[Dict[str, Any]]:
        """Get all transcriptions for a specific course."""
        transcriptions = []
        
        for file_key, info in self.metadata["transcriptions"].items():
            if info.get("course_name") == course_name:
                transcription_path = Path(info.get("transcription_path", ""))
                
                if transcription_path.exists():
                    try:
                        with open(transcription_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Remove header metadata from content
                        lines = content.split('\n')
                        content_start = 0
                        for i, line in enumerate(lines):
                            if not line.startswith('#'):
                                content_start = i
                                break
                        
                        clean_content = '\n'.join(lines[content_start:]).strip()
                        
                        transcriptions.append({
                            "original_file": info.get("original_file"),
                            "transcription_path": str(transcription_path),
                            "content": clean_content,
                            "method": info.get("method"),
                            "created": info.get("created"),
                            "character_count": len(clean_content)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not read transcription {transcription_path}: {e}")
        
        return transcriptions
    
    def should_skip_media_file(self, media_file: Path, course_name: str) -> bool:
        """Determine if media file should be skipped (already transcribed)."""
        return self.has_transcription(media_file, course_name)
    
    def get_media_extensions(self) -> List[str]:
        """Get list of media file extensions that can be transcribed."""
        return ['.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.flac', '.m4a']
    
    def is_media_file(self, file_path: Path) -> bool:
        """Check if file is a media file that can be transcribed."""
        return file_path.suffix.lower() in self.get_media_extensions()
    
    def cleanup_orphaned_transcriptions(self):
        """Remove transcriptions for media files that no longer exist."""
        orphaned_keys = []
        
        for file_key, info in self.metadata["transcriptions"].items():
            original_file = Path(file_key)
            if not original_file.exists():
                orphaned_keys.append(file_key)
                
                # Remove transcription file
                transcription_path = Path(info.get("transcription_path", ""))
                if transcription_path.exists():
                    try:
                        transcription_path.unlink()
                        logger.info(f"Removed orphaned transcription: {transcription_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {transcription_path}: {e}")
        
        # Remove from metadata
        for key in orphaned_keys:
            del self.metadata["transcriptions"][key]
        
        if orphaned_keys:
            self._save_metadata()
            logger.info(f"Cleaned up {len(orphaned_keys)} orphaned transcriptions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics."""
        total_transcriptions = len(self.metadata["transcriptions"])
        total_chars = sum(info.get("character_count", 0) for info in self.metadata["transcriptions"].values())
        
        methods = {}
        for info in self.metadata["transcriptions"].values():
            method = info.get("method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        
        return {
            "total_transcriptions": total_transcriptions,
            "total_characters": total_chars,
            "methods_used": methods,
            "storage_location": str(self.transcriptions_dir)
        }