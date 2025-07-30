#!/usr/bin/env python3
"""
Advanced Path Debugging for Transcription Issues
Comprehensive analysis of file path access problems on Windows
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedPathDebugger:
    """Comprehensive path debugging for Windows transcription issues."""
    
    def __init__(self):
        self.working_dir = os.getcwd()
        logger.info(f"🏠 Working directory: {self.working_dir}")
    
    def analyze_path_issue(self, problematic_path: str) -> dict:
        """Analyze a specific path that's causing issues."""
        logger.info(f"🔍 Analyzing problematic path: {problematic_path}")
        
        analysis = {
            "original_path": problematic_path,
            "working_directory": self.working_dir,
            "path_strategies": [],
            "file_system_info": {},
            "recommendations": []
        }
        
        # Test multiple path resolution strategies
        strategies = [
            ("original", problematic_path),
            ("absolute", os.path.abspath(problematic_path)),
            ("normpath", os.path.normpath(problematic_path)),
            ("pathlib_resolve", str(Path(problematic_path).resolve())),
            ("join_cwd", os.path.join(self.working_dir, problematic_path)),
            ("forward_slash", problematic_path.replace("\\", "/")),
            ("raw_string", repr(problematic_path)[1:-1]),  # Remove quotes from repr
        ]
        
        for strategy_name, test_path in strategies:
            result = self._test_path_access(strategy_name, test_path)
            analysis["path_strategies"].append(result)
            
            if result["accessible"]:
                logger.info(f"✅ WORKING STRATEGY: {strategy_name}")
                analysis["recommendations"].append(f"Use {strategy_name} strategy: {test_path}")
        
        # Analyze file system context
        analysis["file_system_info"] = self._analyze_file_system_context(problematic_path)
        
        # Generate specific recommendations
        analysis["recommendations"].extend(self._generate_recommendations(analysis))
        
        return analysis
    
    def _test_path_access(self, strategy_name: str, test_path: str) -> dict:
        """Test if a path can be accessed using different methods."""
        result = {
            "strategy": strategy_name,
            "path": test_path,
            "exists": False,
            "is_file": False,
            "accessible": False,
            "size_bytes": None,
            "error": None
        }
        
        try:
            # Check existence
            result["exists"] = os.path.exists(test_path)
            
            if result["exists"]:
                result["is_file"] = os.path.isfile(test_path)
                
                if result["is_file"]:
                    # Test actual file access
                    with open(test_path, 'rb') as f:
                        f.read(1024)  # Try to read first KB
                    
                    result["accessible"] = True
                    result["size_bytes"] = os.path.getsize(test_path)
                    
                    logger.info(f"✅ {strategy_name}: {test_path} - ACCESSIBLE ({result['size_bytes']} bytes)")
                else:
                    logger.warning(f"❌ {strategy_name}: {test_path} - EXISTS but not a file")
            else:
                logger.warning(f"❌ {strategy_name}: {test_path} - Does not exist")
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"🚨 {strategy_name}: {test_path} - ERROR: {e}")
        
        return result
    
    def _analyze_file_system_context(self, problematic_path: str) -> dict:
        """Analyze the file system context around the problematic path."""
        context = {
            "path_components": [],
            "parent_directories": [],
            "encoding_issues": None,
            "special_characters": []
        }
        
        try:
            path_obj = Path(problematic_path)
            
            # Analyze path components
            context["path_components"] = list(path_obj.parts)
            
            # Check each parent directory
            current = path_obj
            while current.parent != current:
                parent_info = {
                    "path": str(current.parent),
                    "exists": current.parent.exists(),
                    "accessible": False
                }
                
                if parent_info["exists"]:
                    try:
                        list(current.parent.iterdir())  # Test directory access
                        parent_info["accessible"] = True
                    except Exception as e:
                        parent_info["error"] = str(e)
                
                context["parent_directories"].append(parent_info)
                current = current.parent
            
            # Check for encoding issues
            try:
                problematic_path.encode('ascii')
            except UnicodeEncodeError:
                context["encoding_issues"] = "Contains non-ASCII characters"
            
            # Check for special characters that might cause issues
            special_chars = ['[', ']', '(', ')', '{', '}', '&', '%', '$', '#', '@', '!']
            found_special = [char for char in special_chars if char in problematic_path]
            if found_special:
                context["special_characters"] = found_special
            
        except Exception as e:
            context["analysis_error"] = str(e)
        
        return context
    
    def _generate_recommendations(self, analysis: dict) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        
        # Check if any strategy worked
        working_strategies = [s for s in analysis["path_strategies"] if s["accessible"]]
        
        if not working_strategies:
            recommendations.extend([
                "🚨 CRITICAL: No path strategy can access the file",
                "Check if file actually exists at the expected location",
                "Verify file permissions and ownership",
                "Check for file locking by other processes"
            ])
        
        # Check for encoding issues
        if analysis["file_system_info"].get("encoding_issues"):
            recommendations.append("🔤 Use UTF-8 path encoding or move files to ASCII-only paths")
        
        # Check for special characters
        if analysis["file_system_info"].get("special_characters"):
            special_chars = analysis["file_system_info"]["special_characters"]
            recommendations.append(f"🔣 Consider renaming files to avoid special characters: {special_chars}")
        
        # Check parent directory access
        parent_dirs = analysis["file_system_info"].get("parent_directories", [])
        inaccessible_parents = [d for d in parent_dirs if d["exists"] and not d.get("accessible", True)]
        
        if inaccessible_parents:
            recommendations.append("📁 Some parent directories exist but are not accessible - check permissions")
        
        return recommendations
    
    def debug_transcription_paths(self, base_directory: str = "archived_courses") -> None:
        """Debug common transcription path issues."""
        logger.info(f"🎯 Debugging transcription paths in: {base_directory}")
        
        base_path = Path(base_directory)
        
        if not base_path.exists():
            logger.error(f"❌ Base directory does not exist: {base_path}")
            return
        
        # Find video files with potential path issues
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        problem_patterns = ['[', ']', '(', ')', '&', '%']
        
        logger.info("🔍 Scanning for video files with potential path issues...")
        
        problem_files = []
        total_videos = 0
        
        for video_ext in video_extensions:
            for video_file in base_path.rglob(f"*{video_ext}"):
                total_videos += 1
                
                # Check if path contains problematic patterns
                path_str = str(video_file)
                if any(pattern in path_str for pattern in problem_patterns):
                    problem_files.append(video_file)
        
        logger.info(f"📊 Found {total_videos} video files, {len(problem_files)} with potential path issues")
        
        # Analyze first few problematic files
        for i, problem_file in enumerate(problem_files[:5]):  # Limit to first 5
            logger.info(f"\n🔬 Analyzing problem file {i+1}: {problem_file}")
            analysis = self.analyze_path_issue(str(problem_file))
            
            # Print recommendations
            if analysis["recommendations"]:
                logger.info("💡 Recommendations:")
                for rec in analysis["recommendations"]:
                    logger.info(f"  • {rec}")

if __name__ == "__main__":
    debugger = AdvancedPathDebugger()
    
    # Debug the specific path from the user's error log
    problematic_path = r"archived_courses\udemy The_Complete_Hands-On_Introduction_to_Apache_Airflow_3\4 - Coding Your First Data Pipeline with Airflow\20 - Define a DAG.mp4"
    
    print("🔍 ADVANCED PATH DEBUGGING")
    print("=" * 60)
    
    analysis = debugger.analyze_path_issue(problematic_path)
    
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"Original path: {analysis['original_path']}")
    print(f"Working directory: {analysis['working_directory']}")
    
    print(f"\n📊 PATH STRATEGY RESULTS:")
    for strategy in analysis["path_strategies"]:
        status = "✅ ACCESSIBLE" if strategy["accessible"] else "❌ FAILED"
        print(f"  {strategy['strategy']}: {status}")
        if strategy["error"]:
            print(f"    Error: {strategy['error']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in analysis["recommendations"]:
        print(f"  • {rec}")
    
    # Also run general debugging
    print(f"\n🎯 GENERAL TRANSCRIPTION PATH DEBUGGING")
    print("=" * 60)
    debugger.debug_transcription_paths()