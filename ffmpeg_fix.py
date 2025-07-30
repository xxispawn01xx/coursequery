"""
FFmpeg Installation and Path Fix for Whisper Transcription
Helps resolve FFmpeg dependency issues on Windows systems
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def check_ffmpeg_availability():
    """Check if FFmpeg is available in system PATH."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            logger.info("✅ FFmpeg is available in system PATH")
            return True
        else:
            logger.warning("⚠️ FFmpeg command failed")
            return False
    except FileNotFoundError:
        logger.error("❌ FFmpeg not found in system PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("❌ FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking FFmpeg: {e}")
        return False

def find_local_ffmpeg():
    """Search for FFmpeg in common installation locations."""
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"C:\Users\%USERNAME%\ffmpeg\bin\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser("~/ffmpeg/bin/ffmpeg.exe"),
    ]
    
    for path in common_paths:
        expanded_path = os.path.expandvars(path)
        if os.path.exists(expanded_path):
            logger.info(f"✅ Found FFmpeg at: {expanded_path}")
            return expanded_path
    
    logger.warning("⚠️ No local FFmpeg installation found")
    return None

def add_ffmpeg_to_path(ffmpeg_path):
    """Add FFmpeg directory to system PATH for current session."""
    try:
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        current_path = os.environ.get('PATH', '')
        
        if ffmpeg_dir not in current_path:
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
            logger.info(f"✅ Added FFmpeg directory to PATH: {ffmpeg_dir}")
            return True
        else:
            logger.info("✅ FFmpeg directory already in PATH")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to add FFmpeg to PATH: {e}")
        return False

def setup_ffmpeg_for_whisper():
    """Setup FFmpeg for Whisper transcription."""
    logger.info("🔧 Setting up FFmpeg for Whisper...")
    
    # Check if FFmpeg is already available
    if check_ffmpeg_availability():
        return True
    
    # Try to find local FFmpeg installation
    ffmpeg_path = find_local_ffmpeg()
    if ffmpeg_path:
        if add_ffmpeg_to_path(ffmpeg_path):
            # Verify it's now available
            if check_ffmpeg_availability():
                return True
    
    # Provide installation instructions
    logger.error("❌ FFmpeg setup failed")
    logger.info("💡 FFmpeg Installation Instructions:")
    logger.info("1. Download FFmpeg from: https://ffmpeg.org/download.html#build-windows")
    logger.info("2. Extract to C:\\ffmpeg\\")
    logger.info("3. Add C:\\ffmpeg\\bin to your system PATH")
    logger.info("4. Restart your command prompt/IDE")
    logger.info("5. Verify with: ffmpeg -version")
    
    return False

def get_whisper_audio_alternatives():
    """Get alternative audio processing methods for Whisper."""
    alternatives = []
    
    # Check for librosa
    try:
        import librosa
        alternatives.append("librosa")
        logger.info("✅ librosa available for direct audio loading")
    except ImportError:
        logger.info("❌ librosa not available")
    
    # Check for pydub
    try:
        import pydub
        alternatives.append("pydub")
        logger.info("✅ pydub available for audio conversion")
    except ImportError:
        logger.info("❌ pydub not available")
    
    # Check for soundfile
    try:
        import soundfile
        alternatives.append("soundfile")
        logger.info("✅ soundfile available for audio I/O")
    except ImportError:
        logger.info("❌ soundfile not available")
    
    return alternatives

if __name__ == "__main__":
    # Test FFmpeg setup
    print("🔧 Testing FFmpeg Setup for Whisper")
    print("=" * 50)
    
    success = setup_ffmpeg_for_whisper()
    
    if success:
        print("✅ FFmpeg is ready for Whisper transcription!")
    else:
        print("❌ FFmpeg setup failed. Checking alternatives...")
        alternatives = get_whisper_audio_alternatives()
        
        if alternatives:
            print(f"✅ Alternative audio libraries available: {', '.join(alternatives)}")
            print("💡 Whisper can use these for direct audio loading")
        else:
            print("❌ No alternative audio libraries found")
            print("💡 Install librosa: pip install librosa")