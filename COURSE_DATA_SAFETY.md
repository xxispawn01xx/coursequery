# Course Data Safety & GitHub Sync

## Your Data is Protected

✅ **Your courses are SAFE** - They're excluded from Git sync via `.gitignore`

## What Happens During GitHub Sync

### Files That Sync (Code Only):
- Python application files (`.py`)
- Configuration files 
- Documentation (`.md`)
- Project structure

### Files That DON'T Sync (Your Private Data):
- `archived_courses/` - Your course content
- `indexed_courses/` - Processed course data
- `models/` - Downloaded AI models
- `cache/` - Temporary processing files
- `vectors/` - Vector embeddings
- `transcriptions/` - Audio/video transcripts

## Why This is Safe

1. **Privacy Protection**: Your course content stays on your local machine
2. **No Data Loss**: Moving from `raw_docs` to `archived_courses` was just a rename
3. **GitHub Ignores It**: `.gitignore` prevents accidental uploads
4. **Local Only**: Your course data exists only where you put it

## What You'll See During Sync

When you use GitHub Desktop:
- **Green (+)**: New code files being added
- **Yellow (M)**: Modified application files  
- **No course files shown**: Because they're ignored

## Your Workflow

1. **Course Content**: Stays local in `archived_courses/`
2. **Code Changes**: Sync through GitHub
3. **AI Processing**: Happens locally with your data
4. **Results**: Generated content can be exported as needed

## If You Want to Share Courses

Only if you choose to:
1. Remove specific folders from `.gitignore`
2. Or manually copy/export processed results
3. Your original course content remains protected

Your multi-gigabyte course collection is completely safe and will never accidentally sync to GitHub.