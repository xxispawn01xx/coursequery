# Transcription Troubleshooting Guide

## Issue: "The system cannot find the file specified" (WinError 2)

### Root Cause Identified
The error occurs because:
1. **File validation passes** - the system correctly detects file structure
2. **Whisper fails** - cannot access actual file content for transcription
3. **Path resolution** - the files exist in your course index but not as accessible media files

### Environment Understanding
- **Development Environment**: Shows course structure and file organization
- **Local RTX 3060 Environment**: Where actual transcription should occur with accessible media files

### Solution Steps

#### 1. Verify File Accessibility
```bash
# Check if your course files are actually accessible
python debug_transcription.py
```

#### 2. Install Required Dependencies (Local System)
```bash
# Install Whisper for RTX 3060
pip install openai-whisper

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Check Course Directory Configuration
- Ensure your course directory points to actual H:\ drive location
- Verify media files are not on network drives or inaccessible locations
- Check that file paths don't have permission restrictions

#### 4. Fix Duplicate Directory Structure
If you see paths like:
```
archived_courses\[Course Name]\[Course Name]\video.mp4
```

This indicates a nested folder issue. The fix:
1. Check your course extraction/organization
2. Ensure course folders don't have duplicate nested structures
3. Use the path validation in the app to detect and fix these automatically

#### 5. Test with Known Working Files
Start with files that definitely exist:
```bash
# Test with a simple file first
python -c "
import whisper
model = whisper.load_model('tiny')
result = model.transcribe('path/to/test/file.mp4')
print(result['text'])
"
```

### Expected Behavior by Environment

#### Development Environment (Current)
- ✅ Shows course structure and organization
- ✅ Detects VTT files and media file counts  
- ✅ Provides setup instructions and dependency status
- ❌ Cannot transcribe files (files not physically present)

#### Local RTX 3060 Environment  
- ✅ Full course files accessible on H:\ drive
- ✅ Whisper and PyTorch with CUDA installed
- ✅ Actual transcription processing capability
- ✅ VTT files saved next to original media files

### Verification Commands

#### Check Dependencies
```bash
python -c "import whisper; print('Whisper OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Test File Access
```bash
python -c "
from pathlib import Path
file = Path('your/course/file.mp4')
print(f'Exists: {file.exists()}')
print(f'Size: {file.stat().st_size if file.exists() else 0} bytes')
"
```

#### Check GPU Status
```bash
nvidia-smi
```

### Workflow for Different Environments

#### Development Environment (Testing Interface)
1. Use interface to select courses and review structure
2. Check dependency installation status
3. Preview file organization and VTT detection
4. Export configuration for local environment

#### Local RTX 3060 Environment (Actual Processing)
1. Install dependencies (Whisper + PyTorch CUDA)
2. Ensure course files accessible on H:\ drive
3. Use same interface for actual transcription
4. Monitor GPU usage and processing progress

### Success Indicators

#### Interface Working Correctly
- ✅ Course dropdown shows courses with file counts
- ✅ VTT detection shows existing vs needed transcriptions
- ✅ Dependency status shows installation requirements
- ✅ Path validation detects and reports file access issues

#### Transcription Working Correctly  
- ✅ Whisper model loads successfully on RTX 3060
- ✅ Files process without "file not found" errors
- ✅ VTT files created next to original media files
- ✅ Progress tracking shows successful completions

This troubleshooting guide helps distinguish between interface functionality (working correctly) and actual transcription capability (requires local setup with accessible files).