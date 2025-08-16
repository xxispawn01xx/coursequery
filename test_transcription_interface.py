#!/usr/bin/env python3
"""
Test and validation script for the transcription interface
This helps debug issues and validate the smart course-based transcription system
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_course_detection():
    """Test course detection functionality."""
    print(" Testing Course Detection")
    print("=" * 50)
    
    try:
        from config import Config
        config = Config()
        
        print(f"Raw docs directory: {config.raw_docs_dir}")
        print(f"Directory exists: {config.raw_docs_dir.exists()}")
        
        if config.raw_docs_dir.exists():
            courses = []
            for item in config.raw_docs_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    file_count = len([f for f in item.rglob('*') if f.is_file()])
                    courses.append({
                        'name': item.name,
                        'path': str(item),
                        'file_count': file_count
                    })
            
            print(f"Found {len(courses)} courses:")
            for course in courses[:5]:  # Show first 5
                name = course['name']
                if len(name) > 50:
                    display_name = name[:47] + "..."
                else:
                    display_name = name
                print(f"  - {display_name} ({course['file_count']} files)")
            
            if len(courses) > 5:
                print(f"  ... and {len(courses) - 5} more courses")
                
            return courses
        else:
            print(" Raw docs directory not found")
            return []
            
    except Exception as e:
        print(f" Error testing course detection: {e}")
        return []

def test_media_detection(course_path: str):
    """Test media file detection in a specific course."""
    print(f"\n Testing Media Detection for: {Path(course_path).name}")
    print("=" * 50)
    
    try:
        course_dir = Path(course_path)
        if not course_dir.exists():
            print(f" Course directory not found: {course_path}")
            return
        
        # Count different file types
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
        subtitle_extensions = ['.vtt', '.srt']
        
        media_files = []
        subtitle_files = []
        
        for ext in video_extensions + audio_extensions:
            files = list(course_dir.rglob(f"*{ext}"))
            media_files.extend(files)
            print(f"  {ext}: {len(files)} files")
        
        for ext in subtitle_extensions:
            files = list(course_dir.rglob(f"*{ext}"))
            subtitle_files.extend(files)
            print(f"  {ext}: {len(files)} files")
        
        print(f"\nSummary:")
        print(f"  📹 Total media files: {len(media_files)}")
        print(f" Total subtitle files: {len(subtitle_files)}")
        
        # Check which media files have corresponding subtitles
        media_with_subtitles = 0
        for media_file in media_files:
            vtt_file = media_file.with_suffix('.vtt')
            srt_file = media_file.with_suffix('.srt')
            if vtt_file.exists() or srt_file.exists():
                media_with_subtitles += 1
        
        needs_transcription = len(media_files) - media_with_subtitles
        print(f" Media with subtitles: {media_with_subtitles}")
        print(f"  ⚡ Need transcription: {needs_transcription}")
        
        # Show sample files (first 3 of each type)
        if media_files:
            print(f"\nSample media files:")
            for i, file in enumerate(media_files[:3]):
                rel_path = file.relative_to(course_dir)
                size_mb = file.stat().st_size / (1024*1024) if file.exists() else 0
                print(f"  {i+1}. {rel_path} ({size_mb:.1f}MB)")
        
        if subtitle_files:
            print(f"\nSample subtitle files:")
            for i, file in enumerate(subtitle_files[:3]):
                rel_path = file.relative_to(course_dir)
                size_kb = file.stat().st_size / 1024 if file.exists() else 0
                print(f"  {i+1}. {rel_path} ({size_kb:.1f}KB)")
        
        return {
            'media_count': len(media_files),
            'subtitle_count': len(subtitle_files),
            'needs_transcription': needs_transcription
        }
        
    except Exception as e:
        print(f" Error testing media detection: {e}")
        return None

def test_transcription_manager():
    """Test transcription manager functionality."""
    print(f"\n🤖 Testing Transcription Manager")
    print("=" * 50)
    
    try:
        from transcription_manager import TranscriptionManager
        tm = TranscriptionManager()
        
        # Test stats
        stats = tm.get_stats()
        print("Transcription Manager Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test supported formats
        formats = tm.get_supported_formats()
        print(f"\nSupported formats: {formats}")
        
        return True
        
    except Exception as e:
        print(f" Error testing transcription manager: {e}")
        return False

def test_file_access(file_path: str):
    """Test if a specific file can be accessed."""
    try:
        path = Path(file_path)
        exists = path.exists()
        readable = path.is_file() if exists else False
        size = path.stat().st_size if readable else 0
        
        return {
            'exists': exists,
            'readable': readable,
            'size': size,
            'error': None
        }
    except Exception as e:
        return {
            'exists': False,
            'readable': False,
            'size': 0,
            'error': str(e)
        }

def generate_validation_report():
    """Generate a comprehensive validation report."""
    print("\n Generating Validation Report")
    print("=" * 50)
    
    report = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'environment': 'replit_development',
        'course_detection': None,
        'sample_media_detection': None,
        'transcription_manager': None,
        'recommendations': []
    }
    
    # Test course detection
    courses = test_course_detection()
    report['course_detection'] = {
        'success': len(courses) > 0,
        'course_count': len(courses),
        'sample_courses': [c['name'][:50] + "..." if len(c['name']) > 50 else c['name'] for c in courses[:3]]
    }
    
    # Test media detection on first course
    if courses:
        first_course = courses[0]
        media_stats = test_media_detection(first_course['path'])
        report['sample_media_detection'] = media_stats
    
    # Test transcription manager
    tm_success = test_transcription_manager()
    report['transcription_manager'] = {'success': tm_success}
    
    # Generate recommendations
    if not courses:
        report['recommendations'].append("No courses detected - ensure course directory is properly configured")
    
    if report['sample_media_detection'] and report['sample_media_detection']['media_count'] == 0:
        report['recommendations'].append("No media files found - may be expected in Replit environment")
    
    if not tm_success:
        report['recommendations'].append("Transcription manager not working - check imports and dependencies")
    
    if not report['recommendations']:
        report['recommendations'].append("All tests passed - interface ready for local RTX 3060 testing")
    
    # Save report
    report_file = Path('transcription_validation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Validation report saved to: {report_file}")
    print("\nKey Findings:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    return report

if __name__ == "__main__":
    print("🧪 Transcription Interface Test & Validation")
    print("=" * 60)
    print("This script tests the smart course-based transcription interface")
    print("to ensure it works properly when transferred to your local RTX 3060 system.")
    print()
    
    try:
        report = generate_validation_report()
        
        print(f"\n Summary:")
        print(f"  Course detection: {' ' if report['course_detection']['success'] else ' '}")
        print(f"  Media detection: {' ' if report.get('sample_media_detection') else ' '}")
        print(f"  Transcription manager: {' ' if report['transcription_manager']['success'] else ' '}")
        
        print(f"\n📍 Environment Note:")
        print(f"  This is the Replit development environment.")
        print(f"  Actual transcription should be performed on your local RTX 3060 system")
        print(f"  where course files are accessible and Whisper is installed.")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()