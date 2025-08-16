"""
Completely Offline Movie Summarizer for Batch Processing
Uses local RTX 3060 models: PySceneDetect + Mistral/Llama + Whisper
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Local imports
from enhanced_scene_detector import EnhancedSceneDetector
from config import Config

logger = logging.getLogger(__name__)

class OfflineMovieSummarizer:
    """
    Completely offline movie summarization using local RTX 3060 models.
    No internet connection required after initial setup.
    """
    
    def __init__(self, movies_directory: str = "movies"):
        self.config = Config()
        self.movies_dir = Path(movies_directory)
        self.output_dir = Path("movie_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize local components
        self.scene_detector = EnhancedSceneDetector()
        self.local_llm = self._init_local_llm()
        self.whisper_model = self._init_whisper()
        
        # Movie-optimized settings
        self.movie_settings = {
            'content_detector_threshold': 27.0,  # Optimized for cinematic cuts
            'min_scene_length': 3.0,  # Shorter scenes for movies
            'max_scenes': 150,  # More scenes for feature films
            'detection_method': 'content',  # Best for movies
            'downscale_factor': 1,  # Full quality for movies
            'subtitle_analysis': True  # Include subtitle content if available
        }
        
    def _init_local_llm(self):
        """Initialize local text generation model (Mistral or Llama)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                from transformers.utils.quantization_config import BitsAndBytesConfig
            import torch
            
            # 4-bit quantization for RTX 3060 efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Try Mistral first, fallback to Llama
            model_options = [
                "mistralai/Mistral-7B-Instruct-v0.1",
                "meta-llama/Llama-2-7b-chat-hf",
                "chavinlo/alpaca-native"  # Smaller fallback
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Loading local model: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    logger.info(f" Successfully loaded {model_name} for offline summarization")
                    return {"model": model, "tokenizer": tokenizer, "name": model_name}
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
                    
            logger.error("No local LLM available - offline summarization disabled")
            return None
            
        except ImportError as e:
            logger.error(f"Missing dependencies for local LLM: {e}")
            return None
    
    def _init_whisper(self):
        """Initialize local Whisper for audio analysis"""
        try:
            import whisper
            model = whisper.load_model("medium", device="cuda" if self.config.has_gpu() else "cpu")
            logger.info(" Local Whisper loaded for audio analysis")
            return model
        except ImportError:
            logger.warning("Whisper not available - install with: pip install openai-whisper")
            return None
        except Exception as e:
            logger.warning(f"Whisper initialization failed: {e}")
            return None
    
    def generate_local_summary(self, scene_context: Dict, movie_title: str) -> str:
        """Generate scene summary using local Mistral/Llama model"""
        if not self.local_llm:
            return "Local summarization not available"
        
        prompt = f"""[INST] Analyze this movie scene and provide a concise 2-line summary.

Movie: {movie_title}
Timestamp: {scene_context.get('timestamp', 'Unknown')}
Duration: {scene_context.get('duration', 'Unknown')} seconds
Visual Context: {scene_context.get('visual_description', 'Scene transition detected')}
Audio Context: {scene_context.get('audio_summary', 'No audio analysis')}

Provide:
1. What happens in this scene (action/dialogue)
2. Why this moment is significant (plot/character development)

Format as 2 short lines suitable for subtitles (max 50 chars each):
[/INST]"""

        try:
            import torch
            
            model = self.local_llm["model"]
            tokenizer = self.local_llm["tokenizer"]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part
            generated = response.split("[/INST]")[-1].strip()
            
            # Clean up and format
            lines = generated.split('\n')[:2]  # Take first 2 lines
            summary = '\n'.join([line.strip()[:50] for line in lines if line.strip()])
            
            return summary or "Scene transition detected"
            
        except Exception as e:
            logger.error(f"Local summary generation failed: {e}")
            return "Scene transition detected"
    
    def analyze_audio_content(self, video_path: Path, start_time: float, duration: float) -> str:
        """Extract and analyze audio content around scene transition"""
        if not self.whisper_model:
            return "No audio analysis available"
        
        try:
            import whisper
            
            # Extract audio segment around scene transition
            audio_segment = whisper.load_audio(str(video_path), sr=16000)
            
            # Get 30-second window around transition
            start_sample = int(max(0, start_time - 15) * 16000)
            end_sample = int(min(len(audio_segment), (start_time + 15) * 16000))
            segment = audio_segment[start_sample:end_sample]
            
            # Transcribe segment
            result = self.whisper_model.transcribe(segment, language="en")
            text = result.get("text", "").strip()
            
            # Summarize dialogue/audio content
            if len(text) > 200:
                text = text[:200] + "..."
            
            return text or "No significant dialogue"
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return "No audio analysis available"
    
    def process_single_movie(self, movie_path: Path) -> Dict[str, Any]:
        """Process a single movie file completely offline"""
        logger.info(f" Processing movie: {movie_path.name}")
        
        movie_title = movie_path.stem
        start_time = time.time()
        
        # 1. Scene Detection (PySceneDetect)
        logger.info(" Detecting scenes...")
        try:
            scene_results = self.scene_detector.process_videos(
                [str(movie_path)], 
                settings=self.movie_settings
            )
            scenes = scene_results.get("scenes", [])
            logger.info(f" Detected {len(scenes)} scenes")
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return {"error": str(e), "movie": movie_title}
        
        # 2. Generate summaries for each scene
        logger.info(" Generating scene summaries...")
        scene_summaries = []
        
        for i, scene in enumerate(scenes):
            scene_context = {
                'timestamp': scene.get('start_time', 0),
                'duration': scene.get('duration', 0),
                'visual_description': f"Scene {i+1} transition",
                'audio_summary': self.analyze_audio_content(
                    movie_path, 
                    scene.get('start_time', 0), 
                    scene.get('duration', 5)
                )
            }
            
            summary = self.generate_local_summary(scene_context, movie_title)
            
            scene_summaries.append({
                'scene_number': i + 1,
                'timestamp': f"{int(scene.get('start_time', 0) // 60):02d}:{int(scene.get('start_time', 0) % 60):02d}",
                'start_time': scene.get('start_time', 0),
                'duration': scene.get('duration', 0),
                'summary': summary,
                'screenshot_path': scene.get('screenshot_path', ''),
                'importance': self._calculate_scene_importance(scene, summary)
            })
            
            if i % 10 == 0:
                logger.info(f" Processed {i+1}/{len(scenes)} scenes")
        
        processing_time = time.time() - start_time
        
        # 3. Create movie analysis report
        movie_analysis = {
            'movie_title': movie_title,
            'file_path': str(movie_path),
            'total_scenes': len(scenes),
            'processing_time_minutes': round(processing_time / 60, 2),
            'analysis_date': datetime.now().isoformat(),
            'local_models_used': {
                'scene_detection': 'PySceneDetect',
                'text_generation': self.local_llm["name"] if self.local_llm else "None",
                'audio_analysis': 'Whisper Medium' if self.whisper_model else "None"
            },
            'scenes': scene_summaries,
            'interactive_timeline': self._create_interactive_timeline(scene_summaries)
        }
        
        # 4. Save analysis
        output_file = self.output_dir / f"{movie_title}_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(movie_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Movie analysis saved: {output_file}")
        return movie_analysis
    
    def _calculate_scene_importance(self, scene: Dict, summary: str) -> str:
        """Calculate scene importance based on duration and content"""
        duration = scene.get('duration', 0)
        
        # Long scenes are usually important
        if duration > 300:  # 5+ minutes
            return "high"
        elif duration > 120:  # 2+ minutes  
            return "medium"
        else:
            return "low"
    
    def _create_interactive_timeline(self, scenes: List[Dict]) -> Dict[str, Any]:
        """Create VLC-compatible interactive timeline"""
        return {
            'navigation_points': [
                {
                    'timestamp': scene['start_time'],
                    'display_time': scene['timestamp'],
                    'summary': scene['summary'],
                    'thumbnail': scene.get('screenshot_path', ''),
                    'importance': scene['importance']
                }
                for scene in scenes
            ],
            'vlc_commands': [
                f"vlc://seek:{scene['start_time']}" 
                for scene in scenes
            ],
            'subtitle_format': [
                f"{scene['timestamp']} --> {scene['timestamp']}\n{scene['summary']}\n"
                for scene in scenes
            ]
        }
    
    def batch_process_movies(self, max_movies: Optional[int] = None) -> Dict[str, Any]:
        """Process all movies in directory completely offline"""
        logger.info(f" Starting batch movie processing: {self.movies_dir}")
        
        # Find all movie files
        movie_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}
        movie_files = [
            f for f in self.movies_dir.rglob('*') 
            if f.suffix.lower() in movie_extensions and f.is_file()
        ]
        
        if max_movies:
            movie_files = movie_files[:max_movies]
        
        logger.info(f" Found {len(movie_files)} movies to process")
        
        batch_results = {
            'total_movies': len(movie_files),
            'processed_movies': 0,
            'failed_movies': 0,
            'total_scenes': 0,
            'start_time': datetime.now().isoformat(),
            'results': []
        }
        
        for i, movie_path in enumerate(movie_files):
            logger.info(f" Processing {i+1}/{len(movie_files)}: {movie_path.name}")
            
            try:
                result = self.process_single_movie(movie_path)
                
                if 'error' not in result:
                    batch_results['processed_movies'] += 1
                    batch_results['total_scenes'] += result.get('total_scenes', 0)
                else:
                    batch_results['failed_movies'] += 1
                
                batch_results['results'].append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {movie_path.name}: {e}")
                batch_results['failed_movies'] += 1
                batch_results['results'].append({
                    'error': str(e),
                    'movie': movie_path.name
                })
        
        batch_results['end_time'] = datetime.now().isoformat()
        
        # Save batch report
        batch_file = self.output_dir / f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Batch processing complete: {batch_file}")
        logger.info(f" Success: {batch_results['processed_movies']}/{batch_results['total_movies']} movies")
        logger.info(f" Total scenes detected: {batch_results['total_scenes']}")
        
        return batch_results

def main():
    """Example usage for 400 movie batch processing"""
    
    # Initialize offline movie summarizer
    summarizer = OfflineMovieSummarizer(movies_directory="H:/Movies")  # Your movie directory
    
    # Process all movies (or limit for testing)
    results = summarizer.batch_process_movies(max_movies=5)  # Remove limit for all 400
    
    print(f" Processed {results['processed_movies']} movies")
    print(f" Total scenes: {results['total_scenes']}")
    print(f" Results saved in: movie_analysis/")

if __name__ == "__main__":
    main()