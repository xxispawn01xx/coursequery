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
        
        # Setup FFmpeg path for Windows compatibility
        self._setup_ffmpeg_environment()
        
        # Audio processing method (can be configured)
        self.audio_method = "Auto-detect (Try FFmpeg first, fallback to alternatives)"
    
    def _setup_ffmpeg_environment(self):
        """Setup FFmpeg environment for Windows compatibility."""
        try:
            import os
            import subprocess
            
            # Check if FFmpeg is already available
            try:
                subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, 
                             check=True, 
                             timeout=5)
                logger.info(" FFmpeg already available in PATH")
                return
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.info(" FFmpeg not in PATH, searching for local installation...")
            
            # Common FFmpeg installation paths on Windows
            ffmpeg_paths = [
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                r"C:\Program Files (x86)\ffmpeg\bin",
                os.path.expanduser(r"~\ffmpeg\bin"),
                r"D:\ffmpeg\bin",
                r"E:\ffmpeg\bin",
            ]
            
            # Search for FFmpeg executable
            for path in ffmpeg_paths:
                ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
                if os.path.exists(ffmpeg_exe):
                    logger.info(f" Found FFmpeg at: {ffmpeg_exe}")
                    
                    # Add to PATH for current session
                    current_path = os.environ.get('PATH', '')
                    if path not in current_path:
                        os.environ['PATH'] = path + os.pathsep + current_path
                        logger.info(f" Added FFmpeg to PATH: {path}")
                    
                    # Verify it's now available
                    try:
                        subprocess.run(['ffmpeg', '-version'], 
                                     capture_output=True, 
                                     check=True, 
                                     timeout=5)
                        logger.info(" FFmpeg now available after PATH update")
                        return
                    except Exception:
                        logger.warning(" FFmpeg found but still not accessible")
                        continue
            
            logger.warning(" FFmpeg not found in common locations")
            logger.info(" Install FFmpeg or use librosa fallback for audio processing")
            
        except Exception as e:
            logger.error(f" Error setting up FFmpeg environment: {e}")
        
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
                logger.info(" RTX 3060 detected - optimizing Whisper for GPU")
                
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
            
            logger.info(f" Whisper {model_size} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio/video file using Whisper.
        
        Args:
            audio_path: Path to audio/video file
            language: Language code (optional)
            
        Returns:
            Dict containing transcription result and metadata
        """
        if not self.whisper_model:
            if not self.load_whisper_model():
                raise Exception("Failed to load Whisper model")
        
        try:
            from pathlib import Path
            
            # Validate file exists
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not self.is_supported_format(str(audio_file)):
                raise ValueError(f"Unsupported format: {audio_file.suffix}")
            
            logger.info(f" Transcribing: {audio_file.name}")
            logger.info(f" File size: {audio_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Set transcription options
            options = {
                "language": language,
                "task": "transcribe",
                "fp16": self.device == "cuda",  # Use FP16 on GPU for efficiency
            }
            
            # Perform transcription with FFmpeg fallback handling
            logger.info(f" Starting Whisper transcription on {self.device}...")
            
            # Check audio processing method and handle accordingly
            if self.audio_method == "Force FFmpeg (Fastest, requires FFmpeg installation)":
                # Force FFmpeg usage
                try:
                    result = self.whisper_model.transcribe(str(audio_file), **options)
                except FileNotFoundError as ffmpeg_error:
                    if "ffmpeg" in str(ffmpeg_error).lower() or "WinError 2" in str(ffmpeg_error):
                        logger.error(" FFmpeg not found but Force FFmpeg mode selected")
                        raise Exception("FFmpeg not found. Please install FFmpeg or change audio processing method.")
                    else:
                        raise ffmpeg_error
            
            elif self.audio_method == "Use librosa (Python library, slower but reliable)":
                # Force librosa usage
                result = self._transcribe_with_librosa(audio_file, options)
                
            elif self.audio_method == "Use moviepy (Good for videos, moderate speed)":
                # Force moviepy usage
                result = self._transcribe_with_moviepy(audio_file, options)
                
            elif self.audio_method == "Use pydub (Universal, slowest but most compatible)":
                # Force pydub usage
                result = self._transcribe_with_pydub(audio_file, options)
                
            else:
                # Auto-detect mode - try FFmpeg first, fallback to alternatives
                try:
                    result = self.whisper_model.transcribe(str(audio_file), **options)
                except FileNotFoundError as ffmpeg_error:
                    if "ffmpeg" in str(ffmpeg_error).lower() or "WinError 2" in str(ffmpeg_error):
                        logger.error(" FFmpeg not found or not accessible")
                        logger.info(" Attempting direct audio loading with multiple fallbacks...")
                        
                        # Try multiple audio loading approaches
                        audio_array = None
                        
                        # Method 1: Try librosa
                        try:
                            import librosa
                            import numpy as np
                            
                            logger.info(" Trying librosa for audio loading...")
                            audio_data, sr = librosa.load(str(audio_file), sr=16000)
                            audio_array = np.array(audio_data, dtype=np.float32)
                            logger.info(f" Librosa loading successful: {len(audio_array)} samples at {sr}Hz")
                            
                        except ImportError:
                            logger.info(" librosa not available, trying other methods...")
                        except Exception as librosa_error:
                            logger.error(f" Librosa loading failed: {librosa_error}")
                        
                        # Method 2: Try moviepy if librosa failed
                        if audio_array is None:
                            try:
                                from moviepy.editor import VideoFileClip
                                import numpy as np
                                
                                logger.info(" Trying moviepy for audio extraction...")
                                with VideoFileClip(str(audio_file)) as video:
                                    audio = video.audio
                                    if audio is not None:
                                        # Extract audio as numpy array
                                        audio_array = audio.to_soundarray(fps=16000)
                                        if len(audio_array.shape) > 1:
                                            audio_array = audio_array.mean(axis=1)  # Convert to mono
                                        audio_array = audio_array.astype(np.float32)
                                        logger.info(f" Moviepy loading successful: {len(audio_array)} samples")
                                    else:
                                        raise Exception("No audio track found in video")
                                        
                            except ImportError:
                                logger.info(" moviepy not available, trying other methods...")
                            except Exception as moviepy_error:
                                logger.error(f" Moviepy loading failed: {moviepy_error}")
                        
                        # Method 3: Try pydub if others failed
                        if audio_array is None:
                            try:
                                from pydub import AudioSegment
                                import numpy as np
                                
                                logger.info("🔊 Trying pydub for audio loading...")
                                audio = AudioSegment.from_file(str(audio_file))
                                
                                # Convert to mono and 16kHz
                                audio = audio.set_channels(1).set_frame_rate(16000)
                                
                                # Convert to numpy array
                                audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
                                audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
                                
                                logger.info(f" Pydub loading successful: {len(audio_array)} samples")
                                
                            except ImportError:
                                logger.info(" pydub not available")
                            except Exception as pydub_error:
                                logger.error(f" Pydub loading failed: {pydub_error}")
                        
                        # If we got audio data, try transcribing
                        if audio_array is not None:
                            try:
                                logger.info(" Transcribing with directly loaded audio...")
                                result = self.whisper_model.transcribe(audio_array, **options)
                                logger.info(" Direct audio transcription successful!")
                            except Exception as transcribe_error:
                                logger.error(f" Direct audio transcription failed: {transcribe_error}")
                                raise Exception(f"Direct audio loading successful but transcription failed: {transcribe_error}")
                        else:
                            # All methods failed
                            logger.error(" All audio loading methods failed")
                            logger.info(" Install one of: librosa, moviepy, or pydub for audio processing")
                            logger.info(" Or install FFmpeg: https://ffmpeg.org/download.html")
                            raise Exception("FFmpeg not found and no alternative audio libraries available. Install FFmpeg, librosa, moviepy, or pydub.")
                    else:
                        raise ffmpeg_error
            
            # Clean up GPU memory
            if self.device == "cuda" and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            
            # Process result
            transcription = {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "duration": len(result.get("segments", [])) * 30 if "segments" in result else 0,  # Estimate
                "model": self.model_name,
                "device": self.device
            }
            
            char_count = len(transcription["text"])
            logger.info(f" Transcription complete: {char_count} characters")
            
            if char_count < 10:
                logger.warning(" Very short transcription - might indicate audio issue")
            
            return transcription
            
        except Exception as e:
            logger.error(f" Transcription failed for {audio_path}: {e}")
            logger.error(f" Error type: {type(e).__name__}")
            
            # Additional debugging info
            try:
                from pathlib import Path
                file_path = Path(audio_path)
                logger.error(f" File exists: {file_path.exists()}")
                logger.error(f" File readable: {os.access(audio_path, os.R_OK)}")
                logger.error(f" Full path: {file_path.absolute()}")
            except Exception as debug_error:
                logger.error(f" Debug info error: {debug_error}")
            
            raise
    
    def _resolve_absolute_path(self, audio_path: str) -> str:
        """Resolve audio path to absolute format for Windows compatibility.
        
        Args:
            audio_path: Original path (relative or absolute)
            
        Returns:
            str: Absolute path that Whisper can access
        """
        import os
        
        # Enhanced debugging for path resolution
        logger.info(f" Resolving path: {audio_path}")
        logger.info(f" Current working directory: {os.getcwd()}")
        
        # Convert to Path object for manipulation
        path_obj = Path(audio_path)
        
        # Log original path status
        logger.info(f" Original path exists: {path_obj.exists()}")
        logger.info(f" Original path is file: {path_obj.is_file() if path_obj.exists() else 'Unknown'}")
        
        # Try different path resolution strategies
        strategies = [
            ("absolute", os.path.abspath(audio_path)),
            ("normpath", os.path.normpath(audio_path)),
            ("resolved", str(path_obj.resolve())),
            ("cwd_relative", os.path.join(os.getcwd(), audio_path)),
        ]
        
        # Test each strategy
        for strategy_name, resolved_path in strategies:
            logger.info(f"🧪 Testing {strategy_name}: {resolved_path}")
            
            # Check if file exists and is accessible
            if os.path.exists(resolved_path) and os.path.isfile(resolved_path):
                try:
                    # Test file accessibility by attempting to open
                    with open(resolved_path, 'rb') as f:
                        f.read(1024)  # Read first KB to verify access
                    
                    logger.info(f" {strategy_name} strategy successful: {resolved_path}")
                    return resolved_path
                except Exception as access_error:
                    logger.warning(f" {strategy_name} exists but not accessible: {access_error}")
            else:
                logger.warning(f" {strategy_name} file not found: {resolved_path}")
        
        # If all strategies fail, log comprehensive debugging info
        logger.error(f"🚨 All path resolution strategies failed for: {audio_path}")
        logger.error(f" Working directory: {os.getcwd()}")
        logger.error(f" Original exists check: {os.path.exists(audio_path)}")
        logger.error(f" Original file check: {os.path.isfile(audio_path) if os.path.exists(audio_path) else 'N/A'}")
        
        # Try to provide helpful debugging information
        if os.path.exists(audio_path):
            try:
                stat = os.stat(audio_path)
                logger.error(f" File stats - Size: {stat.st_size} bytes, Mode: {oct(stat.st_mode)}")
            except Exception as stat_error:
                logger.error(f" Cannot get file stats: {stat_error}")
        
        # Return original path as last resort (let Whisper handle the error)
        logger.warning(f" Using original path as fallback: {audio_path}")
        return audio_path
    
    def _format_path_for_whisper(self, resolved_path: str) -> str:
        """Format resolved path specifically for Whisper compatibility on Windows.
        
        Args:
            resolved_path: Absolute path that exists and is accessible
            
        Returns:
            str: Path formatted for Whisper on Windows
        """
        import os
        
        logger.info(f" Formatting path for Whisper: {resolved_path}")
        
        # Try multiple Whisper-specific path formats
        whisper_formats = [
            ("raw_absolute", resolved_path),
            ("forward_slash", resolved_path.replace("\\", "/")),
            ("double_backslash", resolved_path.replace("\\", "\\\\")),
            ("pathlib_str", str(Path(resolved_path))),
            ("os_normpath", os.path.normpath(resolved_path)),
        ]
        
        # Test each format with actual file access
        for format_name, test_path in whisper_formats:
            try:
                logger.info(f"🧪 Testing Whisper format {format_name}: {test_path}")
                
                # Test if the path can be opened (what Whisper will try to do)
                with open(test_path, 'rb') as f:
                    f.read(1024)  # Read first KB to verify access
                
                logger.info(f" Whisper format {format_name} accessible: {test_path}")
                return test_path
                
            except Exception as format_error:
                logger.warning(f" Whisper format {format_name} failed: {format_error}")
                continue
        
        # If all formats fail, try Windows short path as last resort
        try:
            import ctypes
            from ctypes import wintypes
            
            # Windows API to get short path (8.3 format)
            _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            _GetShortPathNameW.restype = wintypes.DWORD
            
            buffer_size = 260
            buffer = ctypes.create_unicode_buffer(buffer_size)
            ret = _GetShortPathNameW(resolved_path, buffer, buffer_size)
            
            if ret:
                short_path = buffer.value
                logger.info(f" Using Windows short path: {short_path}")
                
                # Test short path access
                with open(short_path, 'rb') as f:
                    f.read(1024)
                
                logger.info(f" Windows short path accessible: {short_path}")
                return short_path
                
        except Exception as short_path_error:
            logger.warning(f" Windows short path failed: {short_path_error}")
        
        # Final fallback - return original resolved path
        logger.error(f"🚨 All Whisper path formats failed, using original: {resolved_path}")
        return resolved_path

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
            # Resolve path to absolute format for Windows compatibility
            resolved_path = self._resolve_absolute_path(audio_path)
            
            logger.info(f" Transcribing with Whisper {self.model_name}...")
            logger.info(f" Using resolved path: {resolved_path}")
            
            # RTX 3060 memory optimization
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Final file size check before transcription
            try:
                file_size = os.path.getsize(resolved_path) / (1024 * 1024)  # MB
                logger.info(f" File size: {file_size:.1f} MB")
            except Exception as size_error:
                logger.warning(f" Cannot get file size: {size_error}")
            
            # Additional Whisper path formatting for Windows
            whisper_path = self._format_path_for_whisper(resolved_path)
            logger.info(f" Final Whisper path: {whisper_path}")
            
            # Transcribe with optimizations
            result = self.whisper_model.transcribe(
                whisper_path,
                language=language,
                fp16=self.device == "cuda",  # Use FP16 on GPU for RTX 3060 efficiency
                verbose=False
            )
            
            logger.info(f" Transcription completed - {len(result['text'])} characters")
            
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "model_used": self.model_name,
                "device_used": self.device,
                "file_path": resolved_path
            }
            
        except Exception as e:
            logger.error(f"🚨 Transcription failed for {audio_path}: {e}")
            logger.error(f" Error type: {type(e).__name__}")
            logger.error(f" Current working directory: {os.getcwd()}")
            
            # Additional debugging for file access errors
            if "No such file or directory" in str(e) or "cannot find the file" in str(e):
                logger.error(f" File access error - investigating...")
                
                # Check if it's a path encoding issue
                try:
                    encoded_path = audio_path.encode('utf-8').decode('utf-8')
                    logger.error(f"🔤 UTF-8 encoded path: {encoded_path}")
                except Exception as encoding_error:
                    logger.error(f"🔤 Path encoding issue: {encoding_error}")
            
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
            
            logger.info(f" Saved transcription: {transcription_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            return False
    
    def cleanup_orphaned_transcriptions(self):
        """Clean up orphaned transcription files."""
        # For offline mode, this is a no-op
        pass
    
    def _transcribe_with_librosa(self, audio_file, options):
        """Transcribe using librosa for audio loading."""
        try:
            import librosa
            import numpy as np
            
            logger.info(" Using librosa for audio processing...")
            audio_data, sr = librosa.load(str(audio_file), sr=16000)
            audio_array = np.array(audio_data, dtype=np.float32)
            logger.info(f" Librosa loading successful: {len(audio_array)} samples at {sr}Hz")
            
            return self.whisper_model.transcribe(audio_array, **options)
            
        except ImportError:
            raise Exception("librosa not installed. Install with: pip install librosa")
        except Exception as e:
            raise Exception(f"Librosa audio processing failed: {e}")
    
    def _transcribe_with_moviepy(self, audio_file, options):
        """Transcribe using moviepy for audio extraction."""
        try:
            from moviepy.editor import VideoFileClip
            import numpy as np
            
            logger.info(" Using moviepy for audio processing...")
            with VideoFileClip(str(audio_file)) as video:
                audio = video.audio
                if audio is not None:
                    audio_array = audio.to_soundarray(fps=16000)
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)  # Convert to mono
                    audio_array = audio_array.astype(np.float32)
                    logger.info(f" Moviepy loading successful: {len(audio_array)} samples")
                    
                    return self.whisper_model.transcribe(audio_array, **options)
                else:
                    raise Exception("No audio track found in video")
                    
        except ImportError:
            raise Exception("moviepy not installed. Install with: pip install moviepy")
        except Exception as e:
            raise Exception(f"Moviepy audio processing failed: {e}")
    
    def _transcribe_with_pydub(self, audio_file, options):
        """Transcribe using pydub for audio processing."""
        try:
            from pydub import AudioSegment
            import numpy as np
            
            logger.info("🔊 Using pydub for audio processing...")
            audio = AudioSegment.from_file(str(audio_file))
            
            # Convert to mono and 16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            
            logger.info(f" Pydub loading successful: {len(audio_array)} samples")
            
            return self.whisper_model.transcribe(audio_array, **options)
            
        except ImportError:
            raise Exception("pydub not installed. Install with: pip install pydub")
        except Exception as e:
            raise Exception(f"Pydub audio processing failed: {e}")

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