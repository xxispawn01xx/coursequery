"""
Local Whisper Transcription Manager
Handles offline audio/video transcription using OpenAI Whisper
Optimized for RTX 3060 GPU acceleration
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Try to import torch, fall back gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class WhisperTranscriptionManager:
    """Manages offline Whisper transcription with RTX 3060 optimization."""
    
    def __init__(self):
        """Initialize Whisper transcription manager."""
        self.whisper_model = None
        self.model_name = "base"  # Start with base model for RTX 3060
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
    def load_whisper_model(self, model_size: str = "base") -> bool:
        """Load Whisper model for transcription.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            import whisper
            
            logger.info(f"Loading Whisper {model_size} model on {self.device}...")
            
            # RTX 3060 optimization - start with smaller models
            if self.device == "cuda":
                logger.info("🚀 RTX 3060 detected - optimizing Whisper for GPU")
                
                # Check GPU memory before loading
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    available = total_memory - allocated
                    
                    logger.info(f"RTX 3060 VRAM: {available:.1f}GB available / {total_memory:.1f}GB total")
                    
                    # Adjust model size based on available memory
                    if available < 2.0 and model_size in ["large", "medium"]:
                        logger.warning(f"RTX 3060 low memory ({available:.1f}GB) - using 'base' model instead of '{model_size}'")
                        model_size = "base"
                    elif available < 1.0:
                        logger.warning(f"RTX 3060 very low memory ({available:.1f}GB) - using 'tiny' model")
                        model_size = "tiny"
            
            # Load the model
            self.whisper_model = whisper.load_model(model_size, device=self.device)
            self.model_name = model_size
            
            logger.info(f"✅ Whisper {model_size} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file using local Whisper.
        
        Args:
            audio_path: Path to audio/video file
            language: Language code (optional, auto-detect if None)
            
        Returns:
            Dict containing transcription results
        """
        if not self.whisper_model:
            if not self.load_whisper_model():
                raise RuntimeError("Whisper model not available")
        
        try:
            logger.info(f"Transcribing {audio_path} with Whisper {self.model_name}...")
            
            # RTX 3060 memory optimization
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Transcribe with optimizations
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                fp16=self.device == "cuda",  # Use FP16 on GPU for RTX 3060 efficiency
                verbose=False
            )
            
            logger.info(f"✅ Transcription completed - {len(result['text'])} characters")
            
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "model_used": self.model_name,
                "device_used": self.device
            }
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise
    
    def transcribe_batch(self, audio_files: List[str], language: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Transcribe multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            language: Language code (optional)
            
        Returns:
            Dict mapping file paths to transcription results
        """
        results = {}
        
        for audio_file in audio_files:
            try:
                result = self.transcribe_audio(audio_file, language)
                results[audio_file] = result
                
                # RTX 3060 memory management between files
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file}: {e}")
                results[audio_file] = {"error": str(e)}
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio/video formats."""
        return [
            ".mp3", ".wav", ".flac", ".m4a", ".ogg",
            ".mp4", ".avi", ".mov", ".mkv", ".webm"
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported for transcription."""
        return Path(file_path).suffix.lower() in self.get_supported_formats()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        return {
            "model_loaded": self.whisper_model is not None,
            "model_name": self.model_name,
            "device": self.device,
            "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "supported_formats": self.get_supported_formats()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription manager statistics for UI display."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'torch_available': TORCH_AVAILABLE,
            'whisper_loaded': self.whisper_model is not None,
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'supported_formats': self.get_supported_formats(),
            'status': 'Ready for RTX 3060 transcription' if TORCH_AVAILABLE else 'PyTorch not available - install locally for GPU acceleration',
            # Add missing fields expected by the UI
            'total_transcriptions': 0,  # Default to 0 until actual transcriptions are tracked
            'total_characters': 0,      # Default to 0 until actual transcriptions are tracked
            'methods_used': {},         # Default to empty dict until methods are tracked
            'storage_location': './transcriptions'  # Default storage location for transcriptions
        }
    
    def get_all_transcriptions(self, course_name: str = None) -> List[Dict[str, Any]]:
        """Get all transcriptions, optionally filtered by course name."""
        # Return empty list for now - this would be populated when actual transcriptions are saved
        # In a full implementation, this would read from a transcriptions database/file
        return []
    
    def get_transcriptions_for_course(self, course_name: str) -> List[Dict[str, Any]]:
        """Get transcriptions for a specific course."""
        return self.get_all_transcriptions(course_name)
    
    def has_transcription(self, media_file, course_name: str) -> bool:
        """Check if a transcription already exists for this file."""
        # Check if VTT/SRT file exists alongside the media file
        from pathlib import Path
        media_path = Path(media_file)
        
        # Check for VTT and SRT files with same name
        vtt_file = media_path.with_suffix('.vtt')
        srt_file = media_path.with_suffix('.srt')
        
        return vtt_file.exists() or srt_file.exists()
    
    def save_transcription(self, media_file, course_name: str, transcription: str, method: str) -> bool:
        """Save transcription to file."""
        try:
            from pathlib import Path
            
            # Create transcription file next to original media file
            media_path = Path(media_file)
            transcription_file = media_path.with_suffix('.vtt')
            
            # Create simple VTT format
            vtt_content = f"""WEBVTT

00:00:00.000 --> 99:59:59.999
{transcription}
"""
            
            # Save the transcription
            transcription_file.write_text(vtt_content, encoding='utf-8')
            
            logger.info(f"✅ Saved transcription: {transcription_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            return False
    
    def cleanup_orphaned_transcriptions(self):
        """Clean up orphaned transcription files."""
        # For offline mode, this is a no-op
        pass

# Alias for backward compatibility
TranscriptionManager = WhisperTranscriptionManager

# Global instance for reuse
_whisper_manager = None

def get_whisper_manager() -> WhisperTranscriptionManager:
    """Get global Whisper transcription manager instance."""
    global _whisper_manager
    if _whisper_manager is None:
        _whisper_manager = WhisperTranscriptionManager()
    return _whisper_manager

# Also provide the expected name
def get_transcription_manager() -> WhisperTranscriptionManager:
    """Get transcription manager instance (alias for consistency)."""
    return get_whisper_manager()