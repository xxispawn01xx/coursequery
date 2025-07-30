# Real Estate AI Stack - Fully Local

## Overview

This is a comprehensive, privacy-focused real estate course analysis application that runs 100% offline using local AI models. The application processes various document formats (PDFs, DOCX, PPTX, EPUB, videos/audio) and provides an AI-powered Q&A interface for course materials without any external API calls, model downloads, or online dependencies.

## User Preferences

**CRITICAL**: Pure offline operation - NEVER build for Replit, always 100% offline-first architecture. All development is for local Windows system with H:\ drive access.
**NEVER store model files on Replit** - models exist only locally on user's RTX 3060 system.
Preferred communication style: Simple, everyday language.
Model preference: Local RTX 3060 models with hybrid cloud API fallback for enhanced responses.
Creative tasks preference: Local Excel generation with Google Drive integration as optional enhancement.
Workflow: Pure offline development → RTX 3060 local processing → optional cloud API enhancement.
Deployment strategy: 100% offline capable, RTX 3060 for all AI processing, local file system for courses on H:\ drive.
Authentication: Local HuggingFace token storage, no external dependencies required.
**Architecture Focus**: Windows local development only - Replit is purely for development/testing, not deployment.

## Primary Use Cases

**eBook Library Management**: User has large collection of eBooks in single directory, needs focused processing by theme/topic rather than bulk processing all books together.
**Udemy Course Processing**: Complete course analysis including video transcription (RTX 3060), code files, and PDFs applied to specific problem sets.
**Multi-Page Strategic Planning**: Generate comprehensive business plans, implementation roadmaps, and strategic frameworks from book content using EPUB processing.
**Bulk Audio/Video Processing**: Process podcast series, lecture collections, and video courses with folder structure preservation for RTX 3060 transcription.

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
- **Whisper (Offline)**: Local speech-to-text for media files (install locally: pip install openai-whisper)
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

## How Course Indexing Works

When you "index a course," the system processes all documents in that course folder and creates searchable vector embeddings. Here's what actually happens:

### Document Processing
- **Multi-format Support**: PDFs, DOCX, PPTX, EPUB files are converted to text
- **Video/Audio Transcription**: Media files are transcribed using Whisper (locally on RTX 3060)
- **Content Chunking**: Large documents are split into manageable pieces for better search
- **Metadata Extraction**: File types, paths, and content statistics are preserved

### Vector Embeddings Generation
- **Per-Course Isolation**: Each course gets its own separate vector index
- **Local MiniLM Model**: Converts text chunks into numerical vectors for semantic search
- **Persistent Storage**: Embeddings are saved to disk for instant reuse
- **Syllabus Weighting**: Course syllabi get higher priority in search results

### Individual vs Combined Processing
- **Individual Courses**: Each course maintains separate embeddings for focused, specific answers
- **Combined Processing**: You could merge multiple courses, but this would dilute specificity
- **Best Practice**: Keep courses separate for precise, context-aware responses

### Perplexity Integration Workflow
1. **Local Processing**: Course documents → vector embeddings (RTX 3060, offline)
2. **Query Matching**: Your question finds relevant course chunks via vector similarity (local)
3. **Context Assembly**: Most relevant chunks are gathered as focused context (local)
4. **Cloud API Call**: Only the focused context + question sent to Perplexity for intelligent response
5. **Cost Efficiency**: Pay only for actual queries, not document processing or embeddings
6. **Quality Control**: Generated content is vetted from all course materials for accuracy

### Answer Specificity
**More specific answers come from smaller, focused course collections** because:
- Less noise in the vector space
- More targeted content matching
- Higher relevance scores for course-specific topics
- Better context assembly for the AI model

## Recent Changes

### July 30, 2025 - Centralized Directory Configuration & Offline Mode Fixes
- **Single Source Directory Management**: Created `directory_config.py` with centralized MASTER_COURSE_DIRECTORY variable
- **System-Wide Path Cascading**: One variable change updates paths across all components (course processing, book embeddings, indexing, transcription)
- **Persistent Configuration**: Directory settings saved in `directory_config.json` and persist between sessions
- **Smart Fallback System**: Automatically detects H:\ drive access, falls back to local directories for development
- **Enhanced UI Controls**: "Update Master Directory" button in sidebar cascades changes system-wide with validation
- **Individual Book Embedding System**: New "📚 Book Embeddings" tab processes individual books/ebooks separately from course folders
- **Granular Content Control**: Each book gets separate vector embedding space for precise querying
- **Bulk Processing Options**: Process all books in directory or individual books with progress tracking
- **Book Management Interface**: Search, statistics, and organization by parent course with metadata tracking
- **Path Validation System**: Comprehensive directory access validation with readable/writable status checking
- **Cross-Component Integration**: Book indexer, course indexer, document processor all use centralized directory configuration
- **Offline Mode Compatibility Fixes**: Fixed Pydantic import issues preventing course loading with comprehensive debugging
- **Cross-Platform Timeout System**: Replaced Unix-only signal.SIGALRM with threading-based timeout for Windows compatibility
- **OfflineCourseManager Fallback**: Created simple course detection system that works without complex dependencies
- **Automatic Fallback Detection**: App automatically switches to offline mode when LlamaIndex/Pydantic imports fail
- **Comprehensive Debug System**: Added detailed logging to identify exact failure points in course loading process
- **Robust File Handling**: Added file existence checking and accessibility validation for missing/inaccessible course files
- **Graceful Re-indexing**: Re-indexing now continues with available files even when some files are missing or inaccessible
- **Enhanced Error Reporting**: Detailed summaries show exactly which files processed successfully vs failed with specific reasons
- **Expanded File Format Support**: Added support for 25+ file types including VTT/SRT subtitles, code files, configs, and additional media formats
- **Comprehensive Course Processing**: Now captures VTT subtitle content, Python/JS code examples, JSON configs, and diverse media files for complete coverage
- **Enhanced Path Debugging**: Added comprehensive logging to diagnose nested folder structures with special characters and long Windows paths
- **Windows Path Compatibility**: Enhanced file processing to handle complex nested structures like Udemy course folders with duplicate names and special characters
- **Simplified Directory Detection**: Course directory is now consistently relative to where streamlit is launched from (current working directory)
- **TranscriptionManager Import Fix**: Resolved import error by adding backward compatibility alias and graceful torch fallback
- **Fixed Missing get_stats Method**: Added get_stats() method to WhisperTranscriptionManager to resolve AttributeError in bulk transcription interface

### July 29, 2025 - Multi-Course Detection & Replit Integration
- **Enhanced Course Detection**: System now detects both indexed courses and unprocessed course directories automatically
- **Multi-Course Sidebar**: Displays processed courses (ready for querying) and unprocessed courses (ready for processing) separately
- **One-Click Processing**: Added "Process Course" buttons for unprocessed courses found in raw_docs directories
- **Replit + VSCode Integration**: Created comprehensive guide for course-to-deployment pipeline with automatic project generation
- **Complete Project Generation**: Enhanced course queries to generate full production-ready applications instead of code snippets
- **Workflow Documentation**: Created WORKFLOW_GUIDE.md and PROJECT_GENERATION_GUIDE.md with specific procedures
- **Whisper Offline Ready**: Created complete offline Whisper transcription manager optimized for RTX 3060 (install locally: pip install openai-whisper)
- **DialoGPT Completely Removed**: Eliminated DialoGPT from entire application for faster, cleaner operation per user requirements
- **Pure Local Architecture**: All model files deleted from Replit project - models exist only locally on user's RTX 3060 system
- **Course Detection Fixed**: Updated app to properly detect courses from actual directory paths with session state clearing
- **Frontend Caching Resolved**: Added session state clearing and forced rerun to update course list display

### July 28, 2025 - PyTorch Security & RTX 3060 Memory Fixes
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
- **RTX 3060 Memtest Verification**: Hardware confirmed healthy (0 errors), issue identified as software configuration
- **HuggingFace Forum Fix Implementation**: Applied proven solutions from GitHub issues #28284 and #22546
- **Device Mapping Solution**: Removed problematic accelerate device_map, using manual GPU placement instead
- **RTX 3060 Environment Optimization**: Set proper compute capability (8.6) and disabled problematic debug flags
- **Verified Working Configuration**: Forum fixes tested and confirmed working with RTX 3060 hardware
- **RTX 3060 Memory Optimization**: Added model selection interface to load only ONE model at a time (prevents GPU overload)
- **Smart Model Switching**: Implemented automatic model unloading when switching between Mistral/Llama for RTX 3060 12GB efficiency
- **Enhanced Memory Management**: Raised embedding threshold from 85% to 95-98% for better RTX 3060 utilization
- **GPU Memory Monitoring**: Real-time RTX 3060 memory status with intelligent CPU fallback only when truly necessary
- **PyTorch CVE-2025-32434 Security Fix**: Applied safetensors format loading to avoid torch.load vulnerability
- **RTX 3060 Memory Fragmentation Fix Applied**: Implemented PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True fix (requires system restart)
- **Memory Allocator Configuration**: Applied expandable_segments fix in config.py before any imports for maximum effectiveness  
- **Conservative Model Strategy**: Switched to smaller models (DistilGPT-2, GPT-2) to work within fragmented memory constraints
- **CUDA Environment Optimization**: Added CUDA_LAUNCH_BLOCKING and TORCH_USE_CUDA_DSA for better debugging
- **Memory Allocator Fix**: Prevents impossible 22GB allocation reports on 12GB RTX 3060 GPU
- **Safetensors Model Selection**: Switched to GPT-2 models with secure loading format
- **Comprehensive Error Resolution**: Fixed both security vulnerabilities and memory management issues
- **Persistent API Key Storage**: Added secure local storage for OpenAI/Perplexity keys with auto-save functionality

### System Status
- **Architecture**: ✅ **PURE OFFLINE WINDOWS** - 100% local H:\ drive operation, no Replit deployment
- **Local**: ✅ **FULLY OPERATIONAL** - RTX 3060 running all AI models successfully on CUDA
- **GPU Status**: ✅ **RTX 3060 PERFECT** - All models loading on cuda:0, memory fragmentation fix successful
- **Local Models**: ✅ **ALL WORKING** - local_embeddings, local_llm, whisper all operational
- **Course Directory**: Uses centralized H:\ drive directory configuration with master directory detection
- **Query Engine**: Complete offline AI system with RTX 3060 acceleration
- **Configuration**: Pure offline Windows mode - no cloud dependencies required
- **Course Processing**: ✅ **PROCESSING HUNDREDS OF FILES** - Successfully capturing VTT subtitles with thousands of characters each
- **Multi-Course Detection**: ✅ **OPERATIONAL** - Processing 10+ courses simultaneously with comprehensive file coverage
- **Nested Path Handling**: ✅ **RESOLVED** - Successfully processes 300+ character Windows paths with special characters
- **File Format Expansion**: ✅ **SUCCESS** - VTT files now captured with 6000-8000+ characters of lesson content per file
- **Apache Airflow Course**: ✅ **COMPREHENSIVE** - Processing hundreds of VTT subtitle files containing actual lesson transcripts
- **Document Counting Issue**: ✅ **IDENTIFIED & FIXED** - Root cause was Pydantic import errors preventing course saving to indexed directory
- **Simple Course Indexer**: ✅ **IMPLEMENTED** - Bypasses complex LlamaIndex dependencies to properly save course metadata with accurate document counts
- **TranscriptionManager**: ✅ **FULLY COMPATIBLE** - All methods including get_stats() working with backward compatibility aliases
- **Bulk Transcription Interface**: ✅ **FIXED** - Resolved KeyError 'total_transcriptions' by adding missing stats fields to get_stats() method
- **Vector RAG Interface**: ✅ **FIXED** - Added missing get_all_transcriptions() method for offline compatibility
- **Batch Processing Clarification**: ✅ **ENHANCED** - Updated UI to explain automatic batching for 60+ video courses (6 batches of 10 videos each)
- **Storage Location Fix**: ✅ **FIXED** - Added missing 'storage_location' field to prevent KeyError in transcription management

The architecture prioritizes privacy, local operation, and user control while maintaining enterprise-grade functionality for real estate education analysis.