# Migration Checklist: Moving to New H:\ Directory

## What to Copy from Old Directory

### 📁 **ESSENTIAL - Your Course Data**
```
OLD_LOCATION\archived_courses\
├── [FreeCourseSite.com] Udemy - Apache Airflow The HandsOn Guide\
├── The_Complete_Hands-On_Introduction_to_Apache_Airflow_3\
└── [All your other course folders]
```
**Copy to:** `H:\coursequery\archived_courses\`

### 📊 **VALUABLE - Processed Indexes** 
```
OLD_LOCATION\indexed_courses\
├── course1_metadata.json
├── course2_embeddings.pkl
└── [All processed course data]
```
**Copy to:** `H:\coursequery\indexed_courses\`

### 📚 **OPTIONAL - Book Embeddings**
```
OLD_LOCATION\book_embeddings\
├── book1_embeddings\
└── book2_embeddings\
```
**Copy to:** `H:\coursequery\book_embeddings\`

### 🔑 **IMPORTANT - API Keys & Config**
```
OLD_LOCATION\
├── api_keys.json (if exists)
├── directory_config.json (if exists)
└── .env files (if any)
```

## What NOT to Copy (Auto-Generated)

### ❌ **Skip These - Will Regenerate**
- `models\` - AI models will download fresh
- `cache\` - Temporary cache files
- `temp\` - Temporary processing files
- `.pythonlibs\` - Python dependencies
- `__pycache__\` - Python cache

## Migration Steps

### Step 1: Clone Repo
```cmd
cd H:\
git clone [your-repo-url] coursequery
cd coursequery
```

### Step 2: Copy Essential Data
```cmd
# Copy your course materials
xcopy "OLD_LOCATION\archived_courses\*" "H:\coursequery\archived_courses\" /E /I /H /Y

# Copy processed indexes (saves hours of reprocessing)
xcopy "OLD_LOCATION\indexed_courses\*" "H:\coursequery\indexed_courses\" /E /I /H /Y

# Copy book embeddings (if you have any)
xcopy "OLD_LOCATION\book_embeddings\*" "H:\coursequery\book_embeddings\" /E /I /H /Y
```

### Step 3: Copy Configuration (if exists)
```cmd
copy "OLD_LOCATION\api_keys.json" "H:\coursequery\" 
copy "OLD_LOCATION\directory_config.json" "H:\coursequery\"
```

### Step 4: Verify Setup
1. Run `streamlit run app.py --server.port 5000`
2. Check that courses are detected
3. Verify processed courses show up
4. Test that existing embeddings work

## Time Savings

**With indexed_courses copied:** Instant course access
**Without indexed_courses:** 1-2 hours reprocessing time per course

## Storage Requirements

**Typical sizes:**
- `archived_courses\`: 10-50GB (your actual course videos/PDFs)
- `indexed_courses\`: 100MB-1GB (vector embeddings)
- `book_embeddings\`: 50-500MB (book indexes)

## Verification Commands

```cmd
# Check course detection
dir "H:\coursequery\archived_courses"

# Check processed courses  
dir "H:\coursequery\indexed_courses"

# Verify total size
dir "H:\coursequery" /s
```

## If Something Goes Wrong

1. **Courses not detected**: Check `archived_courses\` path structure
2. **Need to reprocess**: Delete `indexed_courses\` and reindex
3. **API keys missing**: Re-add through the web interface
4. **Path issues**: Verify `ROOT_COURSEQUERY_DIRECTORY = r"H:\coursequery"`