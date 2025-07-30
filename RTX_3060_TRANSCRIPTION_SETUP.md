# RTX 3060 Transcription Setup Guide

## Complete Setup for Local Transcription

### 1. Install Required Dependencies

```bash
# Install PyTorch with CUDA support for RTX 3060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Whisper for local transcription
pip install openai-whisper

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Course Directory Structure

Your courses should be organized like this:
```
H:\Archive Classes\coursequery\archived_courses\
├── [FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\
│   ├── 1 - Introduction\
│   │   ├── 1 - Important Prerequisites.mp4
│   │   ├── 1 - Important Prerequisites.vtt  (if exists)
│   │   └── 2 - The Roadmap.mp4
│   └── 2 - Core Concepts\
└── Other Courses\
```

### 3. Transcription Workflow

1. **Use Smart Course Selection**: 
   - Go to "🎥 Bulk Transcription" tab
   - Select course from dropdown (shows file counts)
   - System auto-detects media and VTT files

2. **Review Analysis**:
   - Shows: "📹 Videos/Audio: X files"
   - Shows: "📝 VTT/Subtitles: Y files" 
   - Shows: "⚡ Need Transcription: Z files"

3. **Configure Settings**:
   - Method: "🖥️ Local Whisper (RTX 3060)"
   - File types: .mp4, .avi, .mov, .mkv, .mp3, .wav
   - Batch size: 10-15 (optimal for RTX 3060 12GB)
   - Skip existing: ✓ (resumes if interrupted)

4. **Start Processing**:
   - Click "🚀 Start Bulk Transcription"
   - Monitor RTX 3060 usage
   - Transcriptions saved as .vtt files next to videos

### 4. RTX 3060 Optimization

**Memory Management:**
- Start with "base" Whisper model (1GB VRAM)
- Upgrade to "medium" if stable (2GB VRAM)
- Batch processing clears GPU memory between files

**Performance Settings:**
- FP16 precision for efficiency
- CUDA acceleration enabled
- Automatic fallback to CPU if needed

### 5. Troubleshooting

**Error: "The system cannot find the file specified"**
- Check course directory path in config
- Verify files are accessible (not on network drive)
- Ensure no duplicate nested folders

**Error: "Whisper not installed"**
```bash
pip install openai-whisper
```

**Error: "CUDA not available"**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Error: "Out of memory"**
- Reduce batch size to 5-8
- Use "tiny" or "base" Whisper model
- Close other GPU applications

### 6. Expected Performance

**RTX 3060 Processing Speed:**
- Tiny model: ~10x real-time
- Base model: ~5x real-time  
- Medium model: ~2x real-time

**For 60-video Apache Airflow course:**
- Estimated time: 2-4 hours (depending on video length)
- Cost: ~$0.50 in electricity vs $180 cloud transcription
- Saves: $179.50 (99.7% savings)

### 7. Quality Settings

**Recommended for Course Content:**
- Model: "base" (good accuracy, fast processing)
- Language: Auto-detect (or specify "en" for English)
- FP16: Enabled (RTX 3060 optimization)

**For Higher Accuracy:**
- Model: "medium" (better accuracy, slower)
- Post-process: Review generated VTT files
- Edit: Use VTT editor for corrections if needed

### 8. Integration with Course System

**After Transcription:**
1. VTT files are automatically included in course indexing
2. Use "📚 Documents" tab to re-index course 
3. Query transcribed content via "🔍 Vector RAG" tab
4. Export embeddings for work environment transfer

**Vector Embeddings Transfer:**
- Embeddings: 5-50MB per course
- Transfer via USB/cloud storage
- Import at work for querying with cloud APIs
- No need to transfer large video files

This setup gives you the complete offline transcription workflow optimized for your RTX 3060 system.