# Directory Setup Guide

## Centralized Directory Configuration System

The system now uses a centralized directory configuration that cascades throughout all components. You can change your course directory in multiple ways.

## Method 1: Automatic Detection (Recommended)

The system automatically detects your master directory:
- **Default Master Directory**: `H:\Archive Classes\coursequery\archived_courses`
- **Automatic Fallback**: Uses local directories if master not accessible
- **No Setup Required**: Works out of the box for most users

## Method 2: Update via Web Interface

### Step 1: Find Your Course Directory Path

1. **Windows Explorer**: Navigate to your course folder
2. **Click the address bar** (shows the path)
3. **Copy the full path** (e.g., `D:\MyEducation\Courses`)

### Step 2: Update Master Directory

1. Open the sidebar in the app
2. Look for "📁 Course Directory" section
3. Click "📝 Set Custom Directory Path" to expand
4. See current master directory displayed
5. Paste your new path in the text field
6. Click "📂 Update Master Directory"
7. System validates path and updates all components automatically

## Method 3: Edit Configuration File

### Direct Configuration Update

1. **Edit `directory_config.py`** (line 34):
   ```python
   self.MASTER_COURSE_DIRECTORY = r"Your\New\Path\Here"
   ```

2. **Or edit `directory_config.json`**:
   ```json
   {
     "master_directory": "Your\\New\\Path\\Here"
   }
   ```

### Example Paths

**Your Local System:**
```
H:\Archive Classes\coursequery\archived_courses
```

**Common Windows Paths:**
```
C:\Users\YourName\Documents\Courses
D:\MBA_Materials\Courses
E:\Education\University_Courses
```

**Mac/Linux Paths:**
```
/Users/yourname/Documents/Courses
/home/yourname/courses
```

### Troubleshooting

- **Path not found**: Make sure the directory actually exists
- **Permission error**: Ensure the app has access to read that directory
- **Special characters**: Avoid spaces in folder names if possible
- **Network drives**: Local drives work better than network locations

The app will automatically detect all course subdirectories once the correct path is set.