# Centralized Directory Configuration

## Single Point of Control

All H:\ drive paths are now controlled by **one variable** in `directory_config.py`:

```python
# =====================================
# MASTER ROOT DIRECTORY CONFIGURATION  
# =====================================
# Single place to change the root coursequery directory
ROOT_COURSEQUERY_DIRECTORY = r"H:\coursequery"
```

## To Change Root Directory

**Step 1:** Edit `directory_config.py`
```python
# Change this line to your desired root path:
ROOT_COURSEQUERY_DIRECTORY = r"H:\your_new_path"
```

**Step 2:** Restart the application
The change will cascade throughout the entire system automatically.

## What Gets Updated Automatically

When you change `ROOT_COURSEQUERY_DIRECTORY`, these paths update automatically:

- **Master course directory**: `H:\coursequery\archived_courses` 
- **Transcription paths**: All video file paths
- **Course detection**: Automatic course discovery
- **Debug scripts**: Path resolution in debug tools
- **Documentation**: Path examples in guides

## Files That Reference the Central Variable

1. `directory_config.py` - **Main configuration**
2. `debug_transcription.py` - Debug path testing
3. `PROCESS_PATHS.md` - Path documentation
4. `MOVE_TO_H_DRIVE_GUIDE.md` - Migration guide

## Benefits

- **Single point of change**: Update one variable, system-wide changes
- **No hardcoded paths**: All references use the central variable  
- **Easy migration**: Move to different drives/folders easily
- **Consistent paths**: No path mismatches across components
- **Future-proof**: Easy to adapt for different environments

## Example Changes

**To move to D: drive:**
```python
ROOT_COURSEQUERY_DIRECTORY = r"D:\coursequery"
```

**To use a different folder name:**
```python
ROOT_COURSEQUERY_DIRECTORY = r"H:\my_courses"
```

**To use a subfolder:**
```python
ROOT_COURSEQUERY_DIRECTORY = r"H:\education\coursequery"
```

The system will automatically create the full path: `[ROOT]\archived_courses\`