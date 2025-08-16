#!/usr/bin/env python3
"""
Debug script to identify and fix transcription issues
For your local RTX 3060 system
"""

import logging
import sys
from pathlib import Path
from directory_config import ROOT_COURSEQUERY_DIRECTORY

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcription_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def debug_specific_error():
    """Debug the specific error you reported."""
    print(" Debugging Specific Transcription Error")
    print("=" * 50)
    
    # The error path from your logs
    error_path = r"archived_courses\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\1 - Introduction\1 - Important Prerequisites.mp4"
    
    print(f"Error path: {error_path}")
    
    # Convert to Path object
    try:
        file_path = Path(error_path)
        print(f"Normalized path: {file_path}")
        print(f"Absolute path: {file_path.absolute()}")
        print(f"Exists: {file_path.exists()}")
        print(f"Is file: {file_path.is_file() if file_path.exists() else 'N/A'}")
        
        # Check parent directories
        parent = file_path.parent
        print(f"Parent directory: {parent}")
        print(f"Parent exists: {parent.exists()}")
        
        if parent.exists():
            print("Files in parent directory:")
            try:
                for item in parent.iterdir():
                    print(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
            except Exception as e:
                print(f"  Error listing directory: {e}")
        
        # Check if this is a path resolution issue
        # Try different path formats
        alternative_paths = [
            Path("archived_courses") / "[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide" / "1 - Introduction" / "1 - Important Prerequisites.mp4",
            Path(ROOT_COURSEQUERY_DIRECTORY) / "archived_courses" / "[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide" / "1 - Introduction" / "1 - Important Prerequisites.mp4"
        ]
        
        print("\nTrying alternative path formats:")
        for i, alt_path in enumerate(alternative_paths, 1):
            print(f"Alternative {i}: {alt_path}")
            print(f"  Exists: {alt_path.exists()}")
            if alt_path.exists():
                print(f" FOUND! This path works")
                break
                
    except Exception as e:
        print(f"Error analyzing path: {e}")

def debug_directory_structure():
    """Debug the actual directory structure."""
    print("\n Debugging Directory Structure")  
    print("=" * 50)
    
    try:
        from config import Config
        config = Config()
        
        print(f"Raw docs directory: {config.raw_docs_dir}")
        print(f"Directory exists: {config.raw_docs_dir.exists()}")
        
        if config.raw_docs_dir.exists():
            print("\nCourse directories:")
            for item in config.raw_docs_dir.iterdir():
                if item.is_dir():
                    print(f" {item.name}")
                    
                    # Check for the specific Apache Airflow course
                    if "Apache Airflow" in item.name:
                        print(f"    └── Checking Apache Airflow course...")
                        try:
                            # Look for video files
                            mp4_files = list(item.rglob("*.mp4"))
                            print(f"    └── Found {len(mp4_files)} MP4 files")
                            
                            if mp4_files:
                                print("    └── Sample MP4 files:")
                                for mp4 in mp4_files[:3]:
                                    rel_path = mp4.relative_to(item)
                                    exists = mp4.exists()
                                    print(f"         📹 {rel_path} (exists: {exists})")
                        except Exception as e:
                            print(f"    └── Error scanning: {e}")
                            
    except Exception as e:
        print(f"Error debugging directory structure: {e}")

def test_whisper_installation():
    """Test if Whisper is properly installed."""
    print("\n🤖 Testing Whisper Installation")
    print("=" * 50)
    
    try:
        import whisper
        print(" Whisper imported successfully")
        
        # Test CUDA availability
        try:
            import torch
            print(f" PyTorch imported successfully")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name()}")
                
                # Check memory
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB")
                
        except ImportError:
            print(" PyTorch not available")
            
        # Test loading a small model
        print("Testing Whisper model loading...")
        try:
            model = whisper.load_model("tiny")
            print(" Whisper tiny model loaded successfully")
            del model  # Free memory
        except Exception as e:
            print(f" Failed to load Whisper model: {e}")
            
    except ImportError:
        print(" Whisper not installed")
        print("Install with: pip install openai-whisper")

def check_transcription_manager():
    """Check transcription manager functionality."""
    print("\n Testing Transcription Manager")
    print("=" * 50)
    
    try:
        from transcription_manager import TranscriptionManager
        tm = TranscriptionManager()
        
        print(" TranscriptionManager imported successfully")
        
        # Test stats
        stats = tm.get_stats()
        print("Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        # Test file existence check
        test_file = Path("test_video.mp4")  # Non-existent file
        has_transcription = tm.has_transcription(test_file, "test_course")
        print(f"has_transcription for non-existent file: {has_transcription}")
        
    except Exception as e:
        print(f" Error testing TranscriptionManager: {e}")
        import traceback
        traceback.print_exc()

def run_comprehensive_debug():
    """Run all debug tests."""
    print("🧪 Comprehensive Transcription Debug")
    print("=" * 60)
    print("Debugging transcription issues on your local RTX 3060 system")
    print()
    
    debug_specific_error()
    debug_directory_structure() 
    test_whisper_installation()
    check_transcription_manager()
    
    print("\n Debug Summary:")
    print("Check the output above for specific issues.")
    print("Common fixes:")
    print("1. Path issues - ensure course directories are accessible")
    print("2. Install Whisper: pip install openai-whisper")
    print("3. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("4. Verify file permissions and accessibility")

if __name__ == "__main__":
    run_comprehensive_debug()