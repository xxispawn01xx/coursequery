# 🏘️ Real Estate AI Stack - Fully Local

A comprehensive, privacy-focused real estate course analysis application that runs 100% locally using Mistral 7B and Whisper models. No external API calls, no censorship, complete privacy.

## ✨ Features

### 🔒 **100% Local Operation**
- **No external API calls** - everything runs on your machine
- **Complete privacy** - your documents never leave your computer
- **No censorship** - local models under your control
- **GPU optimized** - designed for RTX 3060 and similar GPUs

### 📚 **Comprehensive Document Support**
- **PDFs** - Extract text from real estate textbooks and materials
- **DOCX/PPTX** - Process Word documents and PowerPoint presentations
- **EPUB ebooks** - Support for digital textbooks and reference materials
- **Video/Audio** - Automatic transcription using local Whisper
- **Syllabus weighting** - Prioritize syllabus content in responses

### 🧠 **Advanced AI Capabilities**
- **Llama 2 7B** - Primary conversation model with fine-tuning support
- **Mistral 7B** - Fast fallback model for text generation
- **Local Whisper** - GPU-accelerated video/audio transcription
- **Smart indexing** - LlamaIndex with custom embeddings
- **Course isolation** - Separate, focused responses per course
- **Manual control** - User-controlled re-indexing for efficiency

## 🛠️ System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.9+ (3.11 recommended)
- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU**: RTX 3060 or similar (CPU fallback supported)
- **Storage**: 10GB+ free space for models and indexes

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd real-estate-ai-stack
```

### 2. Install Dependencies
```bash
pip install streamlit torch transformers sentence-transformers llama-index
```

### 3. Set up Authentication
```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login with your token
huggingface-cli login
```

### 4. Run Application
```bash
streamlit run app.py --server.port 5000
```

## 🔄 Model Updates

### When New Model Versions Are Released

**Check for Updates:**
- Visit model pages on HuggingFace to check for new versions
- Llama 2: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

**Update Models (Delete Cache):**
```bash
# Delete specific model cache to force re-download
rm -rf ./models/models--meta-llama--Llama-2-7b-chat-hf/
rm -rf ./models/models--mistralai--Mistral-7B-Instruct-v0.1/

# Or delete entire models folder for fresh start
rm -rf ./models/

# Restart app - new versions will download automatically
streamlit run app.py --server.port 5000
```

**Version Control Options:**
```python
# Pin specific version (optional)
model_name = "meta-llama/Llama-2-7b-chat-hf@v1.0"

# Use latest (default behavior)
model_name = "meta-llama/Llama-2-7b-chat-hf"
```

**Performance Comparison:**
- Use the built-in analytics dashboard to compare model versions
- Monitor loading times, response quality, and memory usage
- Keep backups of working model caches if needed

## 📊 Data Persistence

Your data survives code updates:
- **Course files**: `./raw_docs/` 
- **Processed indexes**: `./indexed_courses/`
- **Conversations**: `./conversations/`
- **Downloaded models**: `./models/`
- **Analytics data**: `./metrics/`

Update workflow: Replit → GitHub → Local sync → Browser refresh
