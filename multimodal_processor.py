"""
Multimodal Document Processor
Combines video transcriptions, transition screenshots, and documents
for comprehensive vector embeddings
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Import existing processors
from document_processor import DocumentProcessor
from video_transition_detector import VideoTransitionDetector

logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """Processes multimodal course content: text, audio, video, and visual."""
    
    def __init__(self):
        """Initialize multimodal processor."""
        self.document_processor = DocumentProcessor()
        self.transition_detector = VideoTransitionDetector()
        
    def process_course_comprehensive(self, course_directory: str) -> Dict[str, Any]:
        """
        Process all course content types for complete multimodal embeddings.
        
        Args:
            course_directory: Path to course directory
            
        Returns:
            Combined processing results with all content types
        """
        course_dir = Path(course_directory)
        if not course_dir.exists():
            logger.error(f"Course directory not found: {course_dir}")
            return {}
        
        logger.info(f" Starting comprehensive multimodal processing: {course_dir.name}")
        
        results = {
            'course_name': course_dir.name,
            'processing_date': datetime.now().isoformat(),
            'content_types': {
                'documents': {'count': 0, 'files': []},
                'transcriptions': {'count': 0, 'files': []},
                'video_transitions': {'count': 0, 'videos': 0},
                'total_content_pieces': 0
            },
            'embedding_documents': []
        }
        
        # 1. Process traditional documents (PDFs, DOCX, PPTX, etc.)
        logger.info(" Processing documents...")
        doc_results = self._process_documents(course_dir)
        results['content_types']['documents'] = doc_results
        results['embedding_documents'].extend(doc_results['embedding_docs'])
        
        # 2. Process existing transcriptions (VTT, SRT files)
        logger.info(" Processing existing transcriptions...")
        transcript_results = self._process_transcriptions(course_dir)
        results['content_types']['transcriptions'] = transcript_results
        results['embedding_documents'].extend(transcript_results['embedding_docs'])
        
        # 3. Process video transitions and screenshots
        logger.info(" Processing video transitions...")
        transition_results = self._process_video_transitions(course_dir)
        results['content_types']['video_transitions'] = transition_results
        results['embedding_documents'].extend(transition_results['embedding_docs'])
        
        # 4. Calculate totals
        total_pieces = (
            len(doc_results['embedding_docs']) +
            len(transcript_results['embedding_docs']) +
            len(transition_results['embedding_docs'])
        )
        results['content_types']['total_content_pieces'] = total_pieces
        
        # 5. Save comprehensive results
        self._save_multimodal_results(course_dir, results)
        
        logger.info(f" Multimodal processing complete: {total_pieces} content pieces")
        return results
    
    def _process_documents(self, course_dir: Path) -> Dict[str, Any]:
        """Process traditional documents (PDFs, DOCX, etc.)."""
        document_extensions = {'.pdf', '.docx', '.pptx', '.epub', '.txt', '.md'}
        
        doc_files = []
        for ext in document_extensions:
            doc_files.extend(course_dir.rglob(f"*{ext}"))
        
        embedding_docs = []
        processed_files = []
        
        for doc_file in doc_files:
            try:
                # Use existing document processor
                documents = self.document_processor.process_file(str(doc_file))
                
                for doc in documents:
                    embedding_docs.append({
                        'content': doc.get('content', ''),
                        'metadata': {
                            'source_type': 'document',
                            'file_type': doc.get('file_type', 'unknown'),
                            'filename': doc_file.name,
                            'relative_path': str(doc_file.relative_to(course_dir)),
                            'character_count': len(doc.get('content', '')),
                            'course_name': course_dir.name
                        }
                    })
                
                processed_files.append({
                    'filename': doc_file.name,
                    'path': str(doc_file.relative_to(course_dir)),
                    'type': doc_file.suffix,
                    'documents_extracted': len(documents)
                })
                
                logger.info(f" Processed {doc_file.name}: {len(documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to process {doc_file.name}: {e}")
                continue
        
        return {
            'count': len(processed_files),
            'files': processed_files,
            'embedding_docs': embedding_docs
        }
    
    def _process_transcriptions(self, course_dir: Path) -> Dict[str, Any]:
        """Process existing transcription files (VTT, SRT, TXT)."""
        transcript_extensions = {'.vtt', '.srt', '.txt'}
        
        transcript_files = []
        for ext in transcript_extensions:
            # Look for transcription files
            for file_path in course_dir.rglob(f"*{ext}"):
                # Skip if it's clearly not a transcription
                if any(skip in file_path.name.lower() for skip in ['readme', 'license', 'changelog']):
                    continue
                transcript_files.append(file_path)
        
        embedding_docs = []
        processed_files = []
        
        for transcript_file in transcript_files:
            try:
                # Read transcription content
                content = self._extract_transcript_content(transcript_file)
                
                if content and len(content.strip()) > 50:  # Skip very short files
                    embedding_docs.append({
                        'content': content,
                        'metadata': {
                            'source_type': 'transcription',
                            'file_type': transcript_file.suffix,
                            'filename': transcript_file.name,
                            'relative_path': str(transcript_file.relative_to(course_dir)),
                            'character_count': len(content),
                            'course_name': course_dir.name
                        }
                    })
                    
                    processed_files.append({
                        'filename': transcript_file.name,
                        'path': str(transcript_file.relative_to(course_dir)),
                        'type': transcript_file.suffix,
                        'character_count': len(content)
                    })
                    
                    logger.info(f" Processed {transcript_file.name}: {len(content)} characters")
                
            except Exception as e:
                logger.error(f"Failed to process {transcript_file.name}: {e}")
                continue
        
        return {
            'count': len(processed_files),
            'files': processed_files,
            'embedding_docs': embedding_docs
        }
    
    def _process_video_transitions(self, course_dir: Path) -> Dict[str, Any]:
        """Process video transitions and generate screenshots."""
        # Check if transition analysis already exists
        transition_file = course_dir / "transition_analysis.json"
        
        if transition_file.exists():
            logger.info(" Loading existing transition analysis...")
            with open(transition_file, 'r') as f:
                transition_results = json.load(f)
        else:
            logger.info(" Generating new transition analysis...")
            transition_results = self.transition_detector.process_course_videos(str(course_dir))
        
        # Extract embedding documents from transitions
        embedding_docs = self.transition_detector.get_transition_text_content(transition_results)
        
        return {
            'count': transition_results.get('total_transitions', 0),
            'videos': transition_results.get('total_videos', 0),
            'embedding_docs': embedding_docs
        }
    
    def _extract_transcript_content(self, transcript_file: Path) -> str:
        """Extract clean text from transcript files."""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean VTT/SRT formatting
            if transcript_file.suffix.lower() in ['.vtt', '.srt']:
                # Remove timestamps and formatting
                lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    # Skip timestamps, numbers, and empty lines
                    if (line and 
                        not line.isdigit() and 
                        not '-->' in line and 
                        not line.startswith('WEBVTT') and
                        not line.startswith('NOTE')):
                        lines.append(line)
                
                content = ' '.join(lines)
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract content from {transcript_file}: {e}")
            return ""
    
    def _save_multimodal_results(self, course_dir: Path, results: Dict[str, Any]) -> None:
        """Save comprehensive processing results."""
        results_file = course_dir / "multimodal_analysis.json"
        
        # Create a summary without the actual embedding documents (too large)
        summary = results.copy()
        summary['embedding_documents'] = f"{len(results['embedding_documents'])} documents ready for embedding"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f" Multimodal analysis saved to: {results_file}")
    
    def get_embedding_documents(self, course_directory: str) -> List[Dict[str, Any]]:
        """
        Get all processed documents ready for vector embedding.
        
        Args:
            course_directory: Path to course directory
            
        Returns:
            List of documents formatted for embedding
        """
        results = self.process_course_comprehensive(course_directory)
        return results.get('embedding_documents', [])
    
    def get_content_summary(self, course_directory: str) -> Dict[str, Any]:
        """
        Get summary of available content types in course.
        
        Args:
            course_directory: Path to course directory
            
        Returns:
            Summary of content analysis
        """
        results = self.process_course_comprehensive(course_directory)
        
        return {
            'course_name': results.get('course_name', 'Unknown'),
            'content_types': results.get('content_types', {}),
            'processing_date': results.get('processing_date', ''),
            'ready_for_embedding': len(results.get('embedding_documents', []))
        }

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = MultimodalProcessor()
    
    # Example: Process course comprehensively
    # results = processor.process_course_comprehensive("path/to/course")
    
    # Example: Get embedding documents
    # docs = processor.get_embedding_documents("path/to/course")
    
    print("MultimodalProcessor initialized successfully!")