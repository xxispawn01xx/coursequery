#!/usr/bin/env python3
"""
Course File Analyzer - Shows exactly what files exist in a course directory
and which ones are being processed vs ignored.
"""

import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_course_directory(course_path: Path) -> Dict[str, Any]:
    """Analyze all files in a course directory and categorize them."""
    
    if not course_path.exists():
        logger.error(f"Course directory does not exist: {course_path}")
        return {}
    
    logger.info(f" Analyzing course directory: {course_path}")
    
    # Import document processor to check supported formats
    try:
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        supported_formats = set(processor.get_supported_formats())
        logger.info(f" Supported formats: {len(supported_formats)} types")
    except ImportError as e:
        logger.warning(f"Could not import DocumentProcessor: {e}")
        supported_formats = {'.pdf', '.docx', '.pptx', '.epub', '.mp4', '.avi', '.mov', '.mp3', '.wav'}
    
    # Analyze all files
    all_files = []
    file_types = Counter()
    supported_files = []
    ignored_files = []
    
    total_size = 0
    
    try:
        for file_path in course_path.rglob('*'):
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    file_info = {
                        'path': str(file_path),
                        'name': file_path.name,
                        'extension': file_path.suffix.lower(),
                        'size': file_size,
                        'relative_path': str(file_path.relative_to(course_path))
                    }
                    
                    all_files.append(file_info)
                    file_types[file_path.suffix.lower()] += 1
                    
                    # Check if supported
                    if file_path.suffix.lower() in supported_formats:
                        supported_files.append(file_info)
                    else:
                        ignored_files.append(file_info)
                        
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access file {file_path}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
        return {}
    
    # Create analysis summary
    analysis = {
        'course_path': str(course_path),
        'total_files': len(all_files),
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'supported_files': len(supported_files),
        'ignored_files': len(ignored_files),
        'file_types': dict(file_types),
        'supported_formats': list(supported_formats),
        'supported_file_details': supported_files[:20],  # Show first 20
        'ignored_file_details': ignored_files[:20],  # Show first 20
        'coverage_percentage': round((len(supported_files) / len(all_files)) * 100, 1) if all_files else 0
    }
    
    return analysis

def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print a detailed analysis report."""
    if not analysis:
        print(" No analysis data available")
        return
    
    print(f"\n COURSE FILE ANALYSIS REPORT")
    print(f"" + "="*60)
    print(f" Course: {analysis['course_path']}")
    print(f" Total Files: {analysis['total_files']}")
    print(f" Total Size: {analysis['total_size_mb']} MB")
    print(f" Supported Files: {analysis['supported_files']}")
    print(f" Ignored Files: {analysis['ignored_files']}")
    print(f" Coverage: {analysis['coverage_percentage']}%")
    
    print(f"\n FILE TYPES BREAKDOWN:")
    print("-" * 40)
    for ext, count in sorted(analysis['file_types'].items(), key=lambda x: x[1], reverse=True):
        status = " " if ext in analysis['supported_formats'] else " "
        print(f"{status} {ext:10} : {count:4} files")
    
    print(f"\n SUPPORTED FILE EXAMPLES (showing first 20):")
    print("-" * 50)
    for file_info in analysis['supported_file_details']:
        size_mb = round(file_info['size'] / (1024 * 1024), 2)
        print(f" {file_info['extension']} | {size_mb:6.2f}MB | {file_info['relative_path']}")
    
    if analysis['ignored_files'] > 0:
        print(f"\n IGNORED FILE EXAMPLES (showing first 20):")
        print("-" * 50)
        for file_info in analysis['ignored_file_details']:
            size_mb = round(file_info['size'] / (1024 * 1024), 2)
            print(f"  🚫 {file_info['extension']} | {size_mb:6.2f}MB | {file_info['relative_path']}")
    
    print(f"\n RECOMMENDATIONS:")
    print("-" * 30)
    
    if analysis['coverage_percentage'] < 50:
        print(" LOW COVERAGE: Many files are being ignored")
        
    ignored_extensions = [ext for ext in analysis['file_types'].keys() 
                         if ext not in analysis['supported_formats'] and analysis['file_types'][ext] > 1]
    
    if ignored_extensions:
        print(f" Consider adding support for: {', '.join(ignored_extensions)}")
    
    if analysis['coverage_percentage'] > 80:
        print(" GOOD COVERAGE: Most files are being processed")

def main():
    """Main function to analyze a course directory."""
    try:
        # Get course path from user or use default
        from directory_config import get_master_directory
        master_dir = Path(get_master_directory())
        
        print(f" Available courses in: {master_dir}")
        
        if not master_dir.exists():
            print(f" Master directory not found: {master_dir}")
            return
        
        # List available courses
        courses = []
        try:
            for item in master_dir.iterdir():
                if item.is_dir():
                    courses.append(item)
        except (OSError, PermissionError) as e:
            print(f" Cannot access master directory: {e}")
            return
        
        if not courses:
            print(f" No courses found in: {master_dir}")
            return
        
        print(f"\n Found {len(courses)} courses:")
        for i, course in enumerate(courses[:10], 1):
            print(f"  {i}. {course.name}")
        
        # Analyze first course as example
        if courses:
            course_to_analyze = courses[0]
            print(f"\n Analyzing: {course_to_analyze.name}")
            
            analysis = analyze_course_directory(course_to_analyze)
            print_analysis_report(analysis)
            
            # Save analysis to file
            output_file = Path("course_analysis.json")
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n Analysis saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        print(f" Analysis failed: {e}")

if __name__ == "__main__":
    main()