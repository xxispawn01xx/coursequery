# Real Estate AI Stack - Fully Local

## Overview

The Real Estate AI Stack is an offline-first system designed for comprehensive media processing and analysis. Initially developed for educational course management, it has expanded to include advanced movie summarization capabilities. Its core purpose is to provide AI-powered insights from various media types (videos, documents, audio) with a strong emphasis on local, privacy-preserving operation, leveraging the user's RTX 3060 GPU for accelerated processing. The project aims to offer a robust, offline alternative to cloud-based AI solutions for both educational and entertainment content analysis.

## User Preferences

**CRITICAL**: Pure offline operation - NEVER build for Replit, always 100% offline-first architecture. All development is for local Windows system with H:\ drive access.
**NEVER store model files on Replit** - models exist only locally on user's RTX 3060 system.
**PySceneDetect is OPTIONAL** - System works 100% offline with basic OpenCV detection, PySceneDetect only enhances accuracy when available.
Preferred communication style: Simple, everyday language.
Model preference: Local RTX 3060 models with hybrid cloud API fallback for enhanced responses.
Creative tasks preference: Local Excel generation with Google Drive integration as optional enhancement.
Workflow: Pure offline development → RTX 3060 local processing → optional cloud API enhancement.
Deployment strategy: 100% offline capable, RTX 3060 for all AI processing, local file system for courses on H:\ drive.
Authentication: Local HuggingFace token storage, no external dependencies required.
**Architecture Focus**: Windows local development only - Replit is purely for development/testing, not deployment.

## System Architecture

### Core Design Principles
- **Offline-First**: All primary AI processing and data storage occur locally on the user's system.
- **Modularity**: Python application designed with clear separation of concerns for maintainability and scalability.
- **GPU Acceleration**: Optimized for RTX 3060 for all AI models (transcription, embeddings, generation, scene detection).
- **Privacy-Centric**: No external network calls for core data processing ensure complete data privacy.
- **Universal GPU Support**: Compatible with any CUDA-supported GPU.

### Frontend
- **Streamlit Web Interface**: Intuitive single-page application for file upload, course/movie selection, and AI querying.
- **Interactive Dashboards**: Real-time chat, progress tracking, and visual feedback for various processing tasks.

### Backend
- **Local Model Management**: Self-contained loading and inference of AI models without external dependencies.
- **Document Processing Pipeline**: Multi-stage processing for diverse file formats with metadata extraction.
- **Vector Search Engine**: LlamaIndex-based retrieval system using custom local embeddings for efficient semantic search.
- **Centralized Configuration**: `directory_config.py` manages all file paths, ensuring system-wide consistency.

### Key Features
- **Interactive Movie Summarization**: AI-powered scene detection with clickable timestamps and intelligent summaries.
- **VLC Plugin Integration**: Seamless overlay system for scene navigation and subtitle-style summaries within VLC Media Player.
- **Educational Course Processing**: AI analysis of video transcriptions, code files, and PDFs.
- **Bulk Media Processing**: Efficiently processes large collections of videos with professional scene detection and AI analysis.
- **Multi-format Document Support**: Handles PDF, DOCX, PPTX, EPUB, VTT/SRT, and code files.
- **AI Models**: Local Mistral 7B Instruct for text generation, Whisper for speech-to-text, and MiniLM for embeddings, all optimized for 4-bit quantization on GPU.
- **Data Storage**: LlamaIndex persistent vector indexes, course-isolated data, and organized file system structure.

## External Dependencies

### Core Libraries
- **Streamlit**: Web interface framework.
- **LlamaIndex**: Document indexing and retrieval.
- **Transformers/PyTorch**: Model loading and inference.
- **Whisper**: Audio/video transcription (requires local installation).
- **PySceneDetect**: Professional scene detection for videos (optional, enhances accuracy).
- **Document Libraries**: `PyPDF2`, `python-docx`, `python-pptx`, `ebooklib`, `librosa`, `moviepy`, `pydub` (for audio processing alternatives).

### AI Models (Local Downloads)
- **Mistral 7B Instruct**: Primary language model.
- **Whisper Medium**: Speech recognition model.
- **MiniLM-L6-v2**: Embedding model.

### Cloud API Fallbacks (Optional for Enhanced Responses)
- **OpenAI API**: For intelligent scene analysis and enhanced query responses (only when explicitly used as fallback for creative tasks).
- **Perplexity API**: For targeted queries (cost-efficient, only sends focused context + question).

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only operation supported.
- **Recommended**: 16GB+ RAM, RTX 3060 or similar CUDA-enabled GPU.
- **Storage**: 10GB+ for models and document indexes.