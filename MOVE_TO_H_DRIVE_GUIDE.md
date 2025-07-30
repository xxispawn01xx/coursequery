# Moving Coursequery to H:\ Drive

## Current vs Target Structure

**Previous Location:**
```
H:\Archive Classes\coursequery\
├── archived_courses\
├── indexed_courses\
├── book_embeddings\
└── models\
```

**Target Location:**
```
H:\coursequery\
├── archived_courses\
├── indexed_courses\
├── book_embeddings\
└── models\
```

## Manual Move Steps

### Step 1: Create Target Directory
```cmd
# Open Command Prompt as Administrator
cd H:\
mkdir coursequery
```

### Step 2: Move coursequery Contents
```cmd
# Move the entire coursequery folder contents (if moving from old location)
xcopy "H:\Archive Classes\coursequery\*" "H:\coursequery\" /E /I /H /Y

# Verify the move was successful
dir "H:\coursequery"
```

### Step 3: Update System Configuration
The system will automatically detect the new location at `H:\coursequery\archived_courses` when you restart the application.

## Alternative: Use System Update Function

In the Streamlit interface:
1. Go to the sidebar
2. Find "Update Master Directory" 
3. Enter: `H:\coursequery\archived_courses`
4. Click Update

This will cascade the change throughout the entire system.

## Verification

After moving, the system should show:
```
📁 Master directory: H:\coursequery\archived_courses
📁 Active directory: H:\coursequery\archived_courses
```

## Benefits of H:\ Root Location

1. **Shorter Paths**: Eliminates "Archive Classes" from paths
2. **Better Performance**: Shorter paths reduce Windows API issues
3. **Cleaner Structure**: Coursequery becomes a top-level directory
4. **Easier Access**: Simpler navigation for manual file operations

## Path Impact on Transcription

**Previous Path:**
```
H:\Archive Classes\coursequery\archived_courses\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\...
```

**Current Path:**
```
H:\coursequery\archived_courses\[FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\...
```

This reduces path length by 16 characters, which can help with Windows long path limitations and improve file access reliability.

## Post-Move Actions

1. Restart the Streamlit application
2. Verify courses are detected properly
3. Test transcription with shorter paths
4. Check that all indexed courses are accessible