# Testing Instructions - Local Development

## Issue Resolution Summary

The "Query engine not initialized" error has been fixed. The issues were:

1. **MockLLM Force Usage**: `Settings.llm = None` in `course_indexer.py` was forcing LlamaIndex to use MockLLM instead of real models
2. **Environment Detection**: Config was incorrectly detecting Replit environment and disabling model loading
3. **Import Dependencies**: Missing LlamaIndex package on Replit (expected for local-only app)

## Fixes Applied

✅ **Removed `Settings.llm = None`** - Now uses actual local models  
✅ **Fixed `skip_model_loading`** - Always False for local app  
✅ **Enhanced error handling** - Better query engine initialization  
✅ **Added model recovery** - Recreates query engine if missing  

## Testing Locally

1. **Sync to GitHub** (repository is now 22MB, fast sync)
2. **Pull to local machine** with GitHub Desktop
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app**:
   ```bash
   streamlit run app.py --server.port 5000
   ```
5. **Test Q&A functionality**:
   - Upload a course document
   - Select the course
   - Ask a question
   - Should see real AI responses (not MockLLM)

## Expected Behavior

- **Models load successfully** on first run (may take 5-10 minutes)
- **Query engine initializes** without "not initialized" error
- **Real AI responses** from Mistral 7B (not mock responses)
- **Persistent authentication** via saved HuggingFace token

## Troubleshooting

If you still see "Query engine not initialized":

1. Check logs for specific error messages
2. Verify HuggingFace token is valid
3. Ensure all dependencies are installed
4. Try restarting the application
5. Check if models downloaded correctly to `./models/` directory

## Key Changes Made

- `config.py`: Removed Replit environment detection
- `course_indexer.py`: Removed MockLLM forcing
- `app.py`: Enhanced query engine initialization logic
- `query_engine.py`: Better error handling for model availability

The app should now work correctly for local Q&A functionality.