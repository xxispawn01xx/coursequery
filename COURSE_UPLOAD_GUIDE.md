# How to Upload Your Courses from H:\Archive Classes

Since your courses are located at `H:\Archive Classes` on your local Windows machine, but the app is running on Replit (Linux), you need to bridge this gap. Here are your options:

## Option 1: File Upload Method (Recommended for Testing)

1. **Open the Streamlit app** in your browser
2. **In the sidebar**, look for "📁 Upload Course Directory"
3. **Click "Browse files"** and navigate to `H:\Archive Classes`
4. **Select one course folder** at a time:
   - Open a course folder (e.g., "Python Programming")
   - Select ALL files in that folder (Ctrl+A)
   - Click "Open"
5. **Enter the course name** (e.g., "Python Programming")
6. **Click "💾 Save Course Files"**
7. **Click "🚀 Process [Course Name] Now"** to index it

## Option 2: Local Development (Best Performance)

For full functionality with all your courses:

1. **Clone this Replit project to GitHub**:
   - Click the GitHub icon in Replit
   - Push to a new repository

2. **Clone locally where your courses exist**:
   ```bash
   git clone [your-repo-url]
   cd [repo-name]
   streamlit run app.py --server.port 5000
   ```

3. **The system will automatically detect** `H:\Archive Classes` and show all your courses

## Option 3: Batch Upload Method

For multiple courses quickly:

1. **Create a ZIP file** of each course folder
2. **Extract each ZIP** to the project's `raw_docs` folder
3. **Use the refresh button** to detect them all

## Supported File Types

The system processes:
- **Documents**: PDF, DOCX, PPTX, EPUB
- **Media**: MP4, AVI, MOV, MP3, WAV
- **Archives**: ZIP (will be extracted)

## What Happens After Upload

1. **Files are saved** to the `raw_docs` directory
2. **Processing extracts** text from all documents
3. **Syllabus detection** automatically finds course outlines
4. **Vector indexing** creates searchable embeddings
5. **Ready for AI querying** with your course content

## Workflow Integration

- **Development**: Use Replit for code changes and testing
- **Course Management**: Upload courses as needed
- **Full Functionality**: Run locally for best AI performance
- **Sync**: Changes automatically sync through GitHub

Your course structure will be preserved and all content will be searchable through the AI interface.