# Offline Course Setup Guide

## Current Situation
You're running the application in a Linux environment but your courses are on Windows path `H:\Archive Classes`. Here are the solutions:

## Solution 1: Copy Courses to Local Directory (Recommended)

```bash
# Create symbolic link or copy your courses
mkdir -p raw_docs
cp -r "/path/to/your/courses/"* raw_docs/

# Or if you have access to H drive through WSL/mount:
cp -r "/mnt/h/Archive Classes/"* raw_docs/
```

## Solution 2: Update Path in Config

Edit `config.py` line 31 to point to your actual course location:

```python
# If running in WSL and H: drive is mounted:
user_course_dir = Path("/mnt/h/Archive Classes")

# Or if courses are in a different location:
user_course_dir = Path("/home/youruser/courses")
```

## Solution 3: Use File Upload (Testing)

The app now has a file uploader in the sidebar:
1. Select files from one course folder
2. Enter course name  
3. Click "Save Course Files"
4. Click "Process Course Now"

## Current Status

✅ **Pure Offline Mode**: No Replit dependencies
✅ **AI Models Enabled**: RTX 3060 optimization active
✅ **Course Detection**: Enhanced debugging and validation
✅ **Fallback Directory**: Using `raw_docs` for local courses

## Next Steps

1. Copy your course folders to `raw_docs/` directory
2. Each course should be its own subdirectory
3. Click "🔄 Refresh Course List" to detect them
4. Select and process your courses

The system is now 100% offline and will work with your RTX 3060 for local AI processing.