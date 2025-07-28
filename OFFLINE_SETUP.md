# 🌐 Complete Offline Setup Guide

This guide will help you set up the Local Course AI Assistant for 100% offline operation.

## 📋 Quick Setup Steps

### 1. Install Dependencies (Internet Required Once)
```bash
# Run the automated installer
python install_dependencies.py

# Or install manually
pip install streamlit plotly pandas torch transformers sentence-transformers llama-index whisper PyPDF2 python-docx python-pptx ebooklib beautifulsoup4
```

### 2. Download AI Models (Internet Required Once)
### 1.5 Hugging Face Authentication (Required for Mistral)
```bash
# Create free account at https://huggingface.co
# Request access at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# Get token from https://huggingface.co/settings/tokens
# Login with token:
huggingface-cli login
```

When you first run the app, it will automatically download:
- **Mistral 7B Instruct** (~4GB) - High-quality instruction model
- **Whisper Medium** (~1.5GB) - Audio/video transcription  
- **MiniLM-L6-v2** (~80MB) - Text embeddings

```bash
# Start the app to trigger model downloads
streamlit run app.py --server.port 5000
# App will be available at: http://127.0.0.1:5000
```

### 3. Optional: Use GGUF Models (Faster & Less RAM)

**Step 1: Choose your model**
- **Zephyr 7B**: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF
- **Mistral 7B**: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF  
- **Llama 2 Chat**: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
- **Code Llama**: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF

**Step 2: Download the right quantization**
- `Q4_K_M.gguf` - Best balance (4GB RAM, good quality) **← Recommended**
- `Q5_K_M.gguf` - Higher quality (5GB RAM)
- `Q8_0.gguf` - Highest quality (7GB RAM)

**Step 3: Download and place in ./models/gguf/**
```bash
# Download Zephyr 7B Q4_K_M (recommended for course analysis)
wget -P ./models/gguf/ https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf

# Or download Mistral 7B Q4_K_M (alternative)
wget -P ./models/gguf/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

**Directory structure after download:**
```
./models/gguf/
├── zephyr-7b-beta.Q4_K_M.gguf
└── mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### 4. Disconnect from Internet
Once setup is complete, you can disconnect from the internet. The app will work completely offline!

## 📁 Directory Structure
```
./
├── models/                 # AI models cache
│   ├── gguf/              # Place GGUF models here
│   └── transformers_cache/ # HuggingFace models
├── indexed_courses/        # Course search indexes
├── raw_docs/              # Uploaded documents
├── app.py                 # Main application
└── install_dependencies.py # Setup script
```

## ✅ What Works Offline

- **Document Processing**: PDFs, Word docs, PowerPoints, EPUBs
- **Video/Audio Transcription**: MP4, AVI, MOV, MP3, WAV files
- **AI Question Answering**: Using local Mistral 7B model
- **Course Search**: Vector-based semantic search
- **Data Storage**: All data stays on your machine

## 🔒 Privacy Benefits

- **No External API Calls**: Everything runs locally
- **No Data Sharing**: Your documents never leave your machine
- **Complete Control**: Perfect for sensitive or proprietary content
- **No Censorship**: Local models aren't filtered or restricted

## 🖥️ System Requirements

**Minimum:**
- 8GB RAM
- 15GB free disk space
- Python 3.8+

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU (RTX 3060 or better)
- 25GB+ free disk space
- SSD storage for faster model loading

## 🚀 Performance Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for faster inference
2. **GGUF Models**: Use quantized models for better performance
3. **SSD Storage**: Store models on SSD for faster loading
4. **Close Other Apps**: Free up RAM for model processing

## 🛠️ Troubleshooting

**Models not downloading?**
- Check internet connection
- Ensure sufficient disk space
- Try restarting the app

**Out of memory errors?**
- Use GGUF models (smaller)
- Close other applications
- Reduce batch size in model config

**GPU not detected?**
- Install CUDA toolkit
- Install GPU-enabled PyTorch
- Check GPU compatibility

## 📞 Need Help?

Check the System Status tab in the app for:
- Dependency installation status
- System information
- Detailed setup instructions