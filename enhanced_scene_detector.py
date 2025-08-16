"""
Enhanced Video Scene Detection using PySceneDetect
Professional-grade scene transition detection for educational content analysis
Optimized for RTX 3060 and high-accuracy multimodal processing
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import io
from PIL import Image
import cv2

# PySceneDetect imports
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
    from scenedetect.video_splitter import split_video_ffmpeg
    from scenedetect.frame_timecode import FrameTimecode
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False

# OpenAI for vision analysis (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedSceneDetector:
    """
    Professional scene detection using PySceneDetect algorithms.
    Significantly more accurate than basic OpenCV histogram comparison.
    """
    
    def __init__(self, 
                 detection_method: str = "content",
                 threshold: float = 30.0,
                 min_scene_len: float = 2.0,
                 max_screenshots_per_video: int = 50,
                 downscale_factor: int = 1):
        """
        Initialize enhanced scene detector.
        
        Args:
            detection_method: "content", "threshold", or "adaptive"
            threshold: Detection sensitivity (content: 30.0, threshold: 12.0)
            min_scene_len: Minimum scene length in seconds
            max_screenshots_per_video: Maximum screenshots to extract
            downscale_factor: Downscale factor for performance (1=no downscale)
        """
        if not PYSCENEDETECT_AVAILABLE:
            logger.warning("PySceneDetect not available - falling back to OpenCV basic detection")
            logger.info("Note: This is an offline-first application. PySceneDetect is optional for enhanced accuracy.")
            raise ImportError("PySceneDetect optional enhancement not available")
        
        self.detection_method = detection_method
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.max_screenshots_per_video = max_screenshots_per_video
        self.downscale_factor = downscale_factor
        self.openai_client = None
        
        # Initialize detector based on method
        self.detector = self._create_detector()
        
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
            except Exception as e:
                logger.warning(f"OpenAI client not initialized: {e}")
    
    def _create_detector(self):
        """Create appropriate PySceneDetect detector."""
        if self.detection_method == "content":
            # Most accurate for general content - HSL color space analysis
            return ContentDetector(threshold=self.threshold)
        elif self.detection_method == "threshold":
            # Fast basic detection - good for simple content
            return ThresholdDetector(threshold=self.threshold)
        elif self.detection_method == "adaptive":
            # Best for fast camera movement and complex scenes
            return AdaptiveDetector(adaptive_threshold=self.threshold)
        else:
            logger.warning(f"Unknown detection method: {self.detection_method}, using content")
            return ContentDetector(threshold=self.threshold)
    
    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect scene changes using PySceneDetect algorithms.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of scene transition data with enhanced analysis
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f" PySceneDetect analyzing: {video_path.name}")
        logger.info(f" Method: {self.detection_method}, Threshold: {self.threshold}")
        
        try:
            # Initialize video manager
            video_manager = VideoManager([str(video_path)])
            scene_manager = SceneManager()
            
            # Add detector
            scene_manager.add_detector(self.detector)
            
            # Set downscale factor for performance
            video_manager.set_downscale_factor(self.downscale_factor)
            
            # Start video and detect scenes
            video_manager.start()
            scene_manager.detect_scenes(
                frame_source=video_manager,
                show_progress=False  # Disable progress bar for cleaner logs
            )
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            video_fps = video_manager.get_framerate()
            
            logger.info(f"📹 Video: {video_fps:.1f} FPS")
            logger.info(f" Found {len(scene_list)} scenes with PySceneDetect")
            
            # Process scenes into transition data
            transitions = []
            for i, scene in enumerate(scene_list):
                if i >= self.max_screenshots_per_video:
                    break
                
                # Get scene start time
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                duration = end_time - start_time
                
                # Skip very short scenes
                if duration < self.min_scene_len:
                    continue
                
                # Capture screenshot at scene start
                screenshot_data = self._capture_scene_screenshot(
                    video_path, start_time, i + 1, video_manager
                )
                
                if screenshot_data:
                    transition_data = {
                        'timestamp': start_time,
                        'scene_number': i + 1,
                        'scene_duration': duration,
                        'detection_method': self.detection_method,
                        'frame_number': int(start_time * video_fps),
                        'screenshot': screenshot_data
                    }
                    transitions.append(transition_data)
            
            video_manager.release()
            
            logger.info(f" Processed {len(transitions)} scene transitions")
            return transitions
            
        except Exception as e:
            logger.error(f"PySceneDetect analysis failed: {e}")
            return []
    
    def _capture_scene_screenshot(self, video_path: Path, timestamp: float, 
                                scene_num: int, video_manager) -> Optional[Dict[str, Any]]:
        """Capture screenshot at scene transition using OpenCV."""
        try:
            # Create output directory
            screenshots_dir = video_path.parent / f"{video_path.stem}_scenes"
            screenshots_dir.mkdir(exist_ok=True)
            
            # Open video with OpenCV for screenshot capture
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Seek to timestamp
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not capture frame at {timestamp:.1f}s")
                cap.release()
                return None
            
            # Save screenshot
            screenshot_filename = f"scene_{scene_num:03d}_{timestamp:.1f}s.jpg"
            screenshot_path = screenshots_dir / screenshot_filename
            
            # High quality save
            cv2.imwrite(str(screenshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cap.release()
            
            # Prepare screenshot data
            screenshot_data = {
                'filename': screenshot_filename,
                'path': str(screenshot_path),
                'timestamp': timestamp,
                'scene_number': scene_num,
                'size': frame.shape[:2],  # (height, width)
            }
            
            # Add visual analysis if OpenAI available
            if self.openai_client:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                analysis = self._analyze_screenshot_content(frame_rgb)
                if analysis:
                    screenshot_data['content_analysis'] = analysis
            
            logger.info(f"📸 Screenshot saved: {screenshot_filename}")
            return screenshot_data
            
        except Exception as e:
            logger.error(f"Failed to capture scene screenshot: {e}")
            return None
    
    def _analyze_screenshot_content(self, frame_rgb) -> Optional[str]:
        """Analyze screenshot content using OpenAI Vision API."""
        if not self.openai_client:
            return None
        
        try:
            # Convert to base64
            pil_image = Image.fromarray(frame_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Analyze with OpenAI Vision
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Latest vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this educational video screenshot for searchable content:

1. **Main Content Type**: (slide, code, diagram, video, etc.)
2. **Visible Text**: Extract any readable text, titles, headings
3. **Key Concepts**: Important terms, technologies, or topics shown
4. **Visual Elements**: Charts, graphs, UI elements, code syntax
5. **Teaching Context**: What appears to be explained or demonstrated

Focus on content that would be useful for course search and Q&A."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "high"  # High detail for better text extraction
                                }
                            }
                        ]
                    }
                ],
                max_tokens=400,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            return None
    
    def process_course_videos(self, course_directory: str, 
                            progress_callback=None) -> Dict[str, Any]:
        """
        Process all videos in a course directory using PySceneDetect.
        
        Args:
            course_directory: Path to course directory
            progress_callback: Optional callback for progress updates
            
        Returns:
            Comprehensive scene analysis results
        """
        course_dir = Path(course_directory)
        if not course_dir.exists():
            logger.error(f"Course directory not found: {course_dir}")
            return {}
        
        # Find video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(course_dir.rglob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in: {course_dir}")
            return {}
        
        logger.info(f" PySceneDetect processing {len(video_files)} videos...")
        
        course_results = {
            'course_name': course_dir.name,
            'processing_date': datetime.now().isoformat(),
            'detection_method': self.detection_method,
            'threshold': self.threshold,
            'total_videos': len(video_files),
            'total_scenes': 0,
            'total_screenshots': 0,
            'videos': {}
        }
        
        # Process each video
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"📹 Processing {i}/{len(video_files)}: {video_file.name}")
            
            if progress_callback:
                progress_callback(i, len(video_files), video_file.name)
            
            try:
                scenes = self.detect_scenes(str(video_file))
                
                video_results = {
                    'filename': video_file.name,
                    'relative_path': str(video_file.relative_to(course_dir)),
                    'scenes_count': len(scenes),
                    'scenes': scenes,
                    'processing_status': 'completed'
                }
                
                course_results['videos'][video_file.stem] = video_results
                course_results['total_scenes'] += len(scenes)
                course_results['total_screenshots'] += len([s for s in scenes if s.get('screenshot')])
                
                logger.info(f" {video_file.name}: {len(scenes)} scenes detected")
                
            except Exception as e:
                logger.error(f"Failed to process {video_file.name}: {e}")
                course_results['videos'][video_file.stem] = {
                    'filename': video_file.name,
                    'processing_status': 'failed',
                    'error': str(e)
                }
                continue
        
        # Save comprehensive results
        results_file = course_dir / "pyscenedetect_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(course_results, f, indent=2, default=str)
        
        logger.info(f" PySceneDetect analysis complete:")
        logger.info(f" {course_results['total_scenes']} scenes detected")
        logger.info(f"   📸 {course_results['total_screenshots']} screenshots captured")
        logger.info(f" Results saved to: {results_file}")
        
        return course_results
    
    def get_embeddings_content(self, course_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract content for vector embeddings from PySceneDetect analysis.
        
        Args:
            course_results: Results from process_course_videos
            
        Returns:
            List of documents ready for vector embedding
        """
        documents = []
        
        for video_name, video_data in course_results.get('videos', {}).items():
            if video_data.get('processing_status') != 'completed':
                continue
                
            for scene in video_data.get('scenes', []):
                # Create rich document for each scene
                doc_content = f"""
Video: {video_data['filename']}
Scene #{scene['scene_number']} at {scene['timestamp']:.1f} seconds
Duration: {scene['scene_duration']:.1f} seconds
Detection: {scene['detection_method']} method

Visual Content Analysis:
{scene.get('screenshot', {}).get('content_analysis', 'No visual analysis available')}

Screenshot Location: {scene.get('screenshot', {}).get('path', 'No screenshot')}
Timestamp: {scene['timestamp']:.1f}s
"""
                
                documents.append({
                    'content': doc_content.strip(),
                    'metadata': {
                        'source_type': 'pyscenedetect_scene',
                        'video_file': video_data['filename'],
                        'scene_number': scene['scene_number'],
                        'timestamp': scene['timestamp'],
                        'duration': scene['scene_duration'],
                        'detection_method': scene['detection_method'],
                        'screenshot_path': scene.get('screenshot', {}).get('path'),
                        'course_name': course_results.get('course_name', 'Unknown'),
                        'has_visual_analysis': bool(scene.get('screenshot', {}).get('content_analysis'))
                    }
                })
        
        return documents
    
    @staticmethod
    def get_detection_recommendations(video_type: str = "educational") -> Dict[str, Any]:
        """Get recommended detection settings for different video types."""
        recommendations = {
            "educational": {
                "method": "content",
                "threshold": 30.0,
                "description": "Good for slide-based content with clear scene changes"
            },
            "lecture": {
                "method": "adaptive", 
                "threshold": 25.0,
                "description": "Handles camera movement and lighting changes"
            },
            "demonstration": {
                "method": "content",
                "threshold": 35.0,
                "description": "Higher threshold for subtle content changes"
            },
            "fast_paced": {
                "method": "threshold",
                "threshold": 15.0,
                "description": "Quick detection for rapidly changing content"
            }
        }
        
        return recommendations.get(video_type, recommendations["educational"])

# Testing and example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if PYSCENEDETECT_AVAILABLE:
        # Initialize detector with educational content settings
        detector = EnhancedSceneDetector(
            detection_method="content",  # Most accurate for educational content
            threshold=30.0,             # Good sensitivity for slide changes
            min_scene_len=2.0,          # Minimum 2 seconds per scene
            max_screenshots_per_video=40  # Max 40 screenshots per video
        )
        
        print(" Enhanced Scene Detector with PySceneDetect initialized!")
        print(f" Detection method: {detector.detection_method}")
        print(f" Threshold: {detector.threshold}")
        
        # Show recommendations
        rec = EnhancedSceneDetector.get_detection_recommendations("educational")
        print(f" Recommended settings: {rec}")
        
    else:
        print(" PySceneDetect not available. Install with: pip install scenedetect[opencv]")