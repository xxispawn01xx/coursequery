"""
Video Transition Screenshot Detector
Custom-built transition detection for multimodal vector embeddings
Optimized for educational course content analysis
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
import base64
import io
from PIL import Image

# OpenAI for vision analysis (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class VideoTransitionDetector:
    """Detects scene transitions in educational videos and captures screenshots."""
    
    def __init__(self, 
                 transition_threshold: float = 0.3,
                 min_scene_duration: float = 2.0,
                 max_screenshots_per_video: int = 50):
        """
        Initialize transition detector.
        
        Args:
            transition_threshold: Minimum difference for scene change (0.0-1.0)
            min_scene_duration: Minimum seconds between transitions
            max_screenshots_per_video: Maximum screenshots to extract per video
        """
        self.transition_threshold = transition_threshold
        self.min_scene_duration = min_scene_duration
        self.max_screenshots_per_video = max_screenshots_per_video
        self.openai_client = None
        
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
            except Exception as e:
                logger.warning(f"OpenAI client not initialized: {e}")
    
    def detect_transitions(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect scene transitions in a video and capture screenshots.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of transition data with screenshots and analysis
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f" Analyzing transitions in: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"📹 Video: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        transitions = []
        prev_histogram = None
        frame_number = 0
        last_transition_time = -self.min_scene_duration
        
        # Sample frames for analysis (not every frame to save time)
        sample_interval = max(1, int(fps * 0.5))  # Sample every 0.5 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_number / fps
            
            # Only analyze sampled frames
            if frame_number % sample_interval == 0:
                # Calculate histogram for transition detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_histogram is not None:
                    # Calculate histogram difference
                    correlation = cv2.compareHist(prev_histogram, hist, cv2.HISTCMP_CORREL)
                    difference = 1.0 - correlation
                    
                    # Check for transition
                    is_transition = (
                        difference > self.transition_threshold and
                        current_time - last_transition_time >= self.min_scene_duration
                    )
                    
                    if is_transition and len(transitions) < self.max_screenshots_per_video:
                        logger.info(f" Transition detected at {current_time:.1f}s (diff: {difference:.3f})")
                        
                        # Capture screenshot
                        screenshot_data = self._capture_screenshot(
                            frame, video_path, current_time, len(transitions) + 1
                        )
                        
                        if screenshot_data:
                            transitions.append({
                                'timestamp': current_time,
                                'frame_number': frame_number,
                                'difference_score': difference,
                                'screenshot': screenshot_data
                            })
                            last_transition_time = current_time
                
                prev_histogram = hist
            
            frame_number += 1
        
        cap.release()
        
        logger.info(f" Found {len(transitions)} transitions in {video_path.name}")
        return transitions
    
    def _capture_screenshot(self, frame: np.ndarray, video_path: Path, 
                          timestamp: float, sequence: int) -> Optional[Dict[str, Any]]:
        """Capture and process a screenshot."""
        try:
            # Create output directory for screenshots
            screenshots_dir = video_path.parent / f"{video_path.stem}_transitions"
            screenshots_dir.mkdir(exist_ok=True)
            
            # Save screenshot
            screenshot_filename = f"transition_{sequence:03d}_{timestamp:.1f}s.jpg"
            screenshot_path = screenshots_dir / screenshot_filename
            
            # Convert BGR to RGB for proper color
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save high-quality screenshot
            cv2.imwrite(str(screenshot_path), frame)
            
            # Prepare data for analysis
            screenshot_data = {
                'filename': screenshot_filename,
                'path': str(screenshot_path),
                'timestamp': timestamp,
                'sequence': sequence,
                'size': frame.shape[:2],  # (height, width)
            }
            
            # Add visual analysis if OpenAI available
            if self.openai_client:
                analysis = self._analyze_screenshot_content(frame_rgb)
                if analysis:
                    screenshot_data['content_analysis'] = analysis
            
            return screenshot_data
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    def _analyze_screenshot_content(self, frame_rgb: np.ndarray) -> Optional[str]:
        """Analyze screenshot content using OpenAI Vision."""
        if not self.openai_client:
            return None
        
        try:
            # Convert frame to base64 for OpenAI API
            pil_image = Image.fromarray(frame_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Analyze with OpenAI Vision
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this educational video screenshot. Describe: 1) Main content (slides, diagrams, code, etc.), 2) Text visible on screen, 3) Visual elements (charts, images, UI). Be concise and focus on searchable content."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "low"  # Faster processing for bulk analysis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            return None
    
    def process_course_videos(self, course_directory: str) -> Dict[str, Any]:
        """
        Process all videos in a course directory for transitions.
        
        Args:
            course_directory: Path to course directory
            
        Returns:
            Combined analysis results for all videos
        """
        course_dir = Path(course_directory)
        if not course_dir.exists():
            logger.error(f"Course directory not found: {course_dir}")
            return {}
        
        # Find video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(course_dir.rglob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in: {course_dir}")
            return {}
        
        logger.info(f" Processing {len(video_files)} videos for transitions...")
        
        course_results = {
            'course_name': course_dir.name,
            'processing_date': datetime.now().isoformat(),
            'total_videos': len(video_files),
            'total_transitions': 0,
            'videos': {}
        }
        
        # Process each video
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"📹 Processing {i}/{len(video_files)}: {video_file.name}")
            
            try:
                transitions = self.detect_transitions(str(video_file))
                
                video_results = {
                    'filename': video_file.name,
                    'relative_path': str(video_file.relative_to(course_dir)),
                    'transitions_count': len(transitions),
                    'transitions': transitions
                }
                
                course_results['videos'][video_file.stem] = video_results
                course_results['total_transitions'] += len(transitions)
                
                logger.info(f" {video_file.name}: {len(transitions)} transitions")
                
            except Exception as e:
                logger.error(f"Failed to process {video_file.name}: {e}")
                continue
        
        # Save course analysis results
        results_file = course_dir / "transition_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(course_results, f, indent=2, default=str)
        
        logger.info(f" Course analysis complete: {course_results['total_transitions']} total transitions")
        logger.info(f" Results saved to: {results_file}")
        
        return course_results
    
    def get_transition_text_content(self, course_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract text content from transition analysis for vector embedding.
        
        Args:
            course_results: Results from process_course_videos
            
        Returns:
            List of text documents for embedding
        """
        documents = []
        
        for video_name, video_data in course_results.get('videos', {}).items():
            for transition in video_data.get('transitions', []):
                # Create searchable document from transition
                doc_content = f"""
Video: {video_data['filename']}
Time: {transition['timestamp']:.1f} seconds
Scene Transition #{transition['sequence']}

Visual Content Analysis:
{transition.get('content_analysis', 'No analysis available')}

Screenshot: {transition['screenshot']['filename']}
Location: {transition['screenshot']['path']}
"""
                
                documents.append({
                    'content': doc_content.strip(),
                    'metadata': {
                        'source_type': 'video_transition',
                        'video_file': video_data['filename'],
                        'timestamp': transition['timestamp'],
                        'screenshot_path': transition['screenshot']['path'],
                        'course_name': course_results.get('course_name', 'Unknown')
                    }
                })
        
        return documents

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = VideoTransitionDetector(
        transition_threshold=0.3,  # Adjust sensitivity
        min_scene_duration=3.0,    # Minimum 3 seconds between transitions
        max_screenshots_per_video=30  # Max 30 screenshots per video
    )
    
    # Example: Process single video
    # transitions = detector.detect_transitions("path/to/video.mp4")
    
    # Example: Process entire course
    # results = detector.process_course_videos("path/to/course/directory")
    
    print("VideoTransitionDetector initialized successfully!")