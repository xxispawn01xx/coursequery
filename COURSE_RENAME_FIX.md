# Course Rename Fix Guide

## When Sidebar Stops Loading After Renaming Folders

If you renamed folders inside your course directories and the sidebar isn't loading, here's how to fix it:

### Quick Fix Options

#### Option 1: Use the Sidebar Fix Button
1. Look for "🚨 Fix Sidebar Loading" button in the sidebar
2. Click it to clear course cache and refresh
3. Sidebar should reload with current course structure

#### Option 2: Manual Cache Clear
1. In the sidebar, look for "🔄 Fix Renamed Courses" section
2. If you see renamed course suggestions:
   - Click "✅ Update" to link old index to new name
   - Click "🗑️ Clean" to remove orphaned indexes
3. Click "🔄 Refresh Course List" to rescan directories

#### Option 3: Complete Reset
1. Click "🔄 Refresh Course List" in sidebar
2. If still not working, restart the Streamlit app:
   ```cmd
   # Stop current process (Ctrl+C)
   streamlit run app.py --server.port 5000
   ```

### What Happens When You Rename Folders

**The Problem:**
- Course indexing system stores references to original folder names
- When you rename folders, indexes still point to old names
- Sidebar can't match current folders with stored indexes

**The Solution:**
- Rename detection automatically finds mismatched names
- Smart matching suggests which renamed folder corresponds to which index
- You can update indexes or clean up orphaned ones

### Prevention Tips

1. **Before Renaming:** Note which courses are already indexed
2. **After Renaming:** Use the rename fix tools immediately
3. **Alternative:** Delete indexed courses before renaming, then re-index

### Example Rename Scenarios

**Scenario 1: Course folder renamed**
```
Old: "Real Estate Course v1"
New: "Real Estate Fundamentals"
Fix: Update index to point to new name
```

**Scenario 2: Subfolder renamed inside course**
```
Old: "Course\Section 1 - Basics"
New: "Course\Section 1 - Introduction" 
Fix: Re-index the entire course
```

**Scenario 3: Multiple folder changes**
```
Fix: Use "Clean" to remove all old indexes, then re-process courses
```

### Technical Details

The rename handler:
- Compares current folder names with indexed course names
- Uses similarity matching to suggest correspondence
- Updates metadata files when you confirm matches
- Preserves your processed embeddings when possible

### When to Re-index vs Update

**Update Index:** When only the folder name changed
- Preserves processed embeddings
- Faster than re-indexing
- Good for simple renames

**Re-index Course:** When folder structure changed significantly
- Processes all files again
- Takes longer but ensures accuracy
- Good for major reorganization