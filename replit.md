# Real Estate AI Stack - Fully Local

## Overview

This is a comprehensive, privacy-focused real estate course analysis application that runs 100% offline using local AI models. The application processes various document formats (PDFs, DOCX, PPTX, EPUB, videos/audio) and provides an AI-powered Q&A interface for course materials without any external API calls, model downloads, or online dependencies.

## User Preferences

Preferred communication style: Simple, everyday language.
Model preference: ChatGPT/Perplexity cloud APIs preferred for better response quality and internet access over local models.
Workflow: Replit → GitHub sync → GitHub Desktop → local development with browser refresh.
Deployment strategy: Replit for development (models disabled), local RTX 3060 for Whisper transcription, cloud APIs for querying.
Authentication: Persistent HuggingFace token storage with GUI input for seamless local usage.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application built with Streamlit providing an intuitive UI for document upload, course selection, and AI-powered querying
- **Interactive Dashboard**: Real-time chat interface with course isolation and visual feedback
- **File Upload System**: Multi-format document upload with progress tracking and validation

### Backend Architecture
- **Modular Python Application**: Clean separation of concerns across multiple specialized modules
- **Local Model Management**: Self-contained AI model loading and inference without external dependencies
- **Document Processing Pipeline**: Multi-stage processing for different file formats with metadata extraction
- **Vector Search Engine**: LlamaIndex-based retrieval system with custom embeddings

### Core Components
1. **Main Application** (`app.py`): Streamlit entry point and UI orchestration
2. **Configuration Management** (`config.py`): Centralized settings and directory management
3. **Document Processor** (`document_processor.py`): Multi-format file parsing and content extraction
4. **Local Models** (`local_models.py`): AI model management for Mistral 7B and embeddings
5. **Course Indexer** (`course_indexer.py`): LlamaIndex integration for document vectorization
6. **Query Engine** (`query_engine.py`): RAG implementation combining retrieval and generation

## Key Components

### Document Processing
- **Multi-format Support**: PDF (PyPDF2), DOCX (python-docx), PPTX (python-pptx), EPUB (ebooklib)
- **Media Transcription**: Local Whisper integration for video/audio files
- **Syllabus Weighting**: Special handling for syllabus documents to prioritize content
- **Metadata Extraction**: File type detection and content categorization

### AI Models
- **Mistral 7B Instruct**: Local text generation with 4-bit quantization for GPU efficiency
- **Whisper Medium**: Local speech-to-text for media files
- **MiniLM Embeddings**: Sentence transformer for semantic search
- **GPU Optimization**: RTX 3060 targeting with CPU fallback support

### Data Storage
- **Vector Index**: LlamaIndex persistent storage for document embeddings
- **Course Isolation**: Separate indexes per course for focused responses
- **File System Structure**: Organized directories for raw documents, indexes, and models
- **Pickle Serialization**: Efficient storage of processed data structures

## Data Flow

1. **Document Ingestion**: Users upload files through Streamlit interface
2. **Format Detection**: MIME type analysis and appropriate processor selection
3. **Content Extraction**: Text extraction with format-specific libraries
4. **Preprocessing**: Text cleaning, chunking, and metadata enrichment
5. **Vectorization**: Local embedding generation using MiniLM model
6. **Index Creation**: LlamaIndex vector store creation with course isolation
7. **Query Processing**: User questions processed through retrieval-augmented generation
8. **Response Generation**: Local Mistral 7B generates contextual answers

## External Dependencies

### Core Libraries
- **Streamlit**: Web interface framework
- **LlamaIndex**: Document indexing and retrieval
- **Transformers/PyTorch**: Model loading and inference
- **Whisper**: Audio/video transcription
- **Document Libraries**: PyPDF2, python-docx, python-pptx, ebooklib

### Model Dependencies
- **Mistral 7B Instruct**: Primary language model (4GB+ download)
- **Whisper Medium**: Speech recognition model (1.5GB+ download)
- **MiniLM-L6-v2**: Embedding model (80MB+ download)

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only operation supported
- **Recommended**: 16GB+ RAM, RTX 3060 or similar GPU
- **Storage**: 10GB+ for models and document indexes

## Deployment Strategy

### Local Development
- **Python Environment**: Virtual environment with requirements.txt
- **Model Caching**: Automatic model download and local storage
- **Directory Structure**: Self-organizing file system with configuration management

### Production Deployment
- **Containerization Ready**: Modular architecture supports Docker deployment
- **GPU Support**: CUDA-enabled containers for accelerated inference
- **Data Persistence**: Configurable storage paths for indexes and models
- **Privacy Compliance**: No external network calls ensure complete data privacy

### Scalability Considerations
- **Memory Management**: Efficient model loading with garbage collection
- **Batch Processing**: Support for bulk document processing
- **Index Optimization**: Manual re-indexing controls for performance tuning
- **Resource Monitoring**: GPU/CPU usage tracking and optimization

## Recent Changes

### July 28, 2025 - Enhanced Analytics & Hybrid Query Engine for Optimal Workflow
- **Complete Repository Cleanup**: Eliminated 5.7GB package cache bloat, reduced to 22MB clean codebase
- **Fresh Git History**: Removed bloated commit history (1.48GB), created clean repository for fast sync
- **Persistent Token Storage**: GUI-based HuggingFace token input with secure file persistence
- **Optimized .gitignore**: Comprehensive exclusion of cache directories (.cache/uv, .pythonlibs)
- **Streamlined Workflow**: Development-only mode on Replit, full AI functionality purely local
- **Enhanced GitHub Integration**: Fast repository sync without model/package bloat
- **Documentation**: Created REPOSITORY_MANAGEMENT.md with cleanup procedures and prevention strategies
- **MockLLM Issue Resolution**: Fixed Settings.llm = None that forced MockLLM usage instead of real models
- **Configuration Fix**: Removed Replit environment detection that incorrectly disabled model loading
- **Query Engine Enhancement**: Added robust initialization with proper error handling and fallback logic
- **NodeWithScore Fix**: Resolved "weighted_score" field error by using tuples instead of object modification
- **Local Deployment Confirmed**: Successfully running on RTX 3060 with Llama 2 7B + Mistral 7B models
- **GPU Optimization**: 4-bit quantization working for RTX 3060 12GB memory efficiency
- **CUDA Error Recovery**: Fixed "device-side assert" errors with graceful fallback instead of crashes
- **Excel Generation**: Automated spreadsheet creation for business valuation and financial analysis
- **Analytics Fix**: Resolved "total_chunks" error in course analytics dashboard
- **Enhanced Error Handling**: CUDA cache clearing, memory management, and timeout protection
- **Business Valuation Ready**: Excel templates for DCF, financial analysis, and valuation frameworks
- **Hybrid Query Engine**: Graceful fallback from local models to cloud APIs for optimal user experience
- **Enhanced Analytics**: Real content analysis replacing placeholder visualizations with meaningful insights
- **Smart Workflow Guidance**: Clear direction to RTX 3060 transcription + ChatGPT Plus querying approach
- **Vector RAG System**: Complete implementation for $400+/year savings with local embeddings + cloud API optimization
- **Bulk Transcription Interface**: Dedicated tab for folder structure preservation and RTX 3060 optimization
- **Cost-Efficient Querying**: Vector similarity search with pay-per-token vs flat-rate subscription savings
- **Clear Billing Interface**: Added explicit billing warnings and tab labeling to distinguish free vs paid features
- **Comprehensive Course Detection**: Fixed analytics to properly detect indexed courses like "vcpe" with 29 documents
- **Enhanced User Clarity**: Overview section explaining exact billing structure and which tabs charge money
- **Comprehensive Vector RAG**: Fixed to process ALL course materials (PDFs, DOCX, PPTX, transcriptions, indexed docs) not just transcriptions
- **Multi-Source Processing**: Vector embeddings now combine documents, transcriptions, and indexed materials for complete course coverage
- **Enhanced Interface**: Clear display of document counts, transcription counts, and comprehensive source breakdown after processing
- **Debug and Error Handling**: Added comprehensive error reporting to show exactly what content sources are being checked
- **Multi-Environment Support**: System works in both Replit (development) and local environments with actual course content
- **Complete Documentation**: Clear explanation of how Vector RAG processes all course materials for intelligent querying
- **GPU Health Testing**: Added comprehensive GPU memory test for RTX 3060 to diagnose potential hardware issues with used graphics card
- **CUDA Debugging Enhanced**: Implemented TORCH_USE_CUDA_DSA for better device-side assert error reporting
- **Performance Monitoring**: Clear visual warnings when CPU fallback is used instead of GPU for embedding generation
- **RTX 3060 Hardware Diagnosis**: Comprehensive testing revealed faulty GPU memory causing device-side assert errors
- **CPU-Only Fallback System**: Implemented automatic detection and graceful CPU-only mode when GPU issues detected
- **Hardware Test Suite**: Created rtx_3060_memory_test.py for systematic GPU health verification
- **Persistent Status Tracking**: GPU health results saved to gpu_test_results.txt for reliable system memory
- **Enhanced Error Classification**: System now distinguishes between hardware issues vs software configuration problems
- **CPU Model Optimization**: Implemented reliable CPU models (DialoGPT, GPT-2, DistilGPT-2) for consistent operation

### System Status
- **Replit**: Clean development environment, ALL ONLINE OPERATIONS DISABLED (offline-only app)
- **Local**: ⚠️ **RTX 3060 HARDWARE ISSUES DETECTED** - GPU has faulty memory causing device-side assert errors
- **CPU Fallback**: ✅ **FULLY OPERATIONAL** - CPU-only mode with DialoGPT, GPT-2, and embedding models
- **Repository**: Successfully cleaned from 5.7GB to 22MB, optimized for fast GitHub sync
- **Query Engine**: Fixed NodeWithScore weighted_score error, now processes queries without crashes
- **Configuration**: Offline-only mode enforced on Replit, full functionality confirmed local
- **Repository Management**: Integrated enforce-planning.md with @REPOSITORY_MANAGEMENT.md requirements
- **Hardware Diagnosis**: Created comprehensive RTX 3060 memory test confirming CUDA unavailability
- **Model Loading**: ✅ Confirmed working locally with CPU-optimized models (slower but reliable)

The architecture prioritizes privacy, local operation, and user control while maintaining enterprise-grade functionality for real estate education analysis.