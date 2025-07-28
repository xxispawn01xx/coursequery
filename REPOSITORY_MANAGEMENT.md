# Repository Management Guide

## Overview

This document explains how we solved the massive repository bloat issue (5.7GB → 22MB) and how to prevent it from happening again.

## The Problem

The repository grew from 22MB of actual code to 5.7GB due to:
- **Package cache bloat**: `.cache/uv/` directory containing 5+ GB of CUDA libraries
- **Bloated Git history**: 1.48GB of packed commit history with old package files
- **Automatic dependency installation**: Replit's UV package manager caching everything locally

## The Solution

### 1. Immediate Cleanup
```bash
# Removed the massive UV cache
rm -rf .cache/uv

# Killed the bloated Git history
rm -rf .git
git init
git add .
git commit -m "Clean repository - local AI course assistant"

# Fresh push to GitHub
git remote add origin https://github.com/xxispawn01xx/coursequery.git
git push --force origin main
```

**Result**: Repository went from 5.7GB to 22MB (99.6% reduction)

### 2. Prevention Strategy

#### Enhanced .gitignore
Our `.gitignore` now comprehensively excludes all cache and package directories:

```gitignore
# Package manager caches (should only exist locally)
.cache/
.uv/
.pythonlibs/
node_modules/
uv.lock

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Local token storage
.hf_token

# Temporary files
temp/
*.tmp
*.log
```

#### Architecture Decision
- **Replit**: Clean development environment, packages auto-install but never commit
- **GitHub**: Only source code (22MB), fast sync, no package bloat
- **Local**: Full functionality with packages that download automatically

## What Caused The Bloat

### UV Package Manager Cache
- UV (Replit's package manager) caches all dependencies in `.cache/uv/`
- CUDA libraries for PyTorch: 2-3GB each
- Multiple versions and architectures cached locally
- Not excluded from Git initially

### Git History Accumulation
- Each commit that included packages grew the `.git` folder
- Packed objects contained 1.48GB of historical package files
- Even after deleting files, Git history preserved everything

### Automatic Dependency Installation
The `install_dependencies.py` script automatically installs:
- PyTorch with CUDA support (2GB+)
- Transformers library (500MB+)
- LlamaIndex dependencies (300MB+)
- All their transitive dependencies

## Prevention Rules

### 1. Never Commit Large Files
- **Packages**: Always local-only installation
- **Models**: Download locally, never push to Git
- **Caches**: Exclude all cache directories

### 2. Regular Repository Health Checks
```bash
# Check repository size
du -sh . --exclude=.git

# Should always be under 50MB for this project

# Check what's being tracked by Git
git ls-files | xargs du -ch | tail -10
```

### 3. Cache Directory Management
```bash
# Safe cleanup when repository grows
rm -rf .cache/uv
rm -rf .pythonlibs
du -sh . --exclude=.git  # Verify size reduction
```

### 4. Deployment Strategy
- **Development**: Replit for coding, no runtime needed
- **Production**: Local machines with full AI capabilities
- **Sync**: GitHub for clean code transfer only

## Repository Health Monitoring

### Size Warnings
- **Green**: Under 25MB (normal source code)
- **Yellow**: 25-100MB (investigate large files)
- **Red**: Over 100MB (immediate cleanup needed)

### What to Track
```bash
# Good files to commit
*.py          # Source code
*.md          # Documentation  
*.toml        # Configuration
*.gitignore   # Git exclusions

# Never commit
.cache/       # Package caches
.pythonlibs/  # Local Python libraries
uv.lock       # Package lock files
*.pkl         # Model files
models/       # AI model directories
```

## Emergency Cleanup Procedure

If the repository grows large again:

### 1. Immediate Actions
```bash
# Stop any running processes
pkill -f python
pkill -f streamlit

# Remove cache directories
rm -rf .cache
rm -rf .pythonlibs
rm -f uv.lock

# Check size
du -sh . --exclude=.git
```

### 2. If Still Large (Git History Bloat)
```bash
# Nuclear option: Fresh Git history
rm -rf .git
git init
git add .
git commit -m "Clean repository reset"
git remote add origin https://github.com/xxispawn01xx/coursequery.git
git push --force origin main
```

### 3. Verify Success
- Repository should be under 25MB
- GitHub shows clean file structure
- All source code preserved

## Best Practices

### For Developers
1. **Check .gitignore before first commit**: Ensure all cache directories excluded
2. **Regular size monitoring**: Run `du -sh . --exclude=.git` weekly
3. **Clean development**: Keep packages local, code in Git
4. **Branch carefully**: Don't accidentally commit package directories

### For Repository Maintenance
1. **Update .gitignore proactively**: Add new cache patterns as they appear
2. **Monitor Git LFS usage**: Large files should use Git LFS if absolutely necessary
3. **Regular cleanup**: Automated or manual cache cleanup in CI/CD

## Success Metrics

- **Repository size**: Maintained under 25MB consistently
- **Clone time**: Under 30 seconds for full repository
- **Sync speed**: Fast GitHub Desktop synchronization
- **Storage efficiency**: 99.6% space savings achieved

## Lessons Learned

1. **Prevention > Cleanup**: Proper .gitignore from start saves hours of cleanup
2. **Package isolation**: Keep development dependencies local, not in version control
3. **Git history matters**: Even deleted files live forever in Git history
4. **Size monitoring**: Regular checks prevent gradual bloat accumulation
5. **Architecture alignment**: Repository strategy must match deployment model

This cleanup enabled the optimal workflow: clean development on Replit, fast GitHub sync, and full AI functionality locally.