# Hugging Face Authentication Setup

## Step 1: Get Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token" 
3. Name: "course-ai-local"
4. Type: **"Read"** (that's all you need)
5. Click "Generate a token"
6. Copy the token (starts with `hf_...`)

## Step 2: Set Up Authentication in Replit

### Option A: Shell Login (Recommended)
```bash
# In Shell tab at bottom of Replit:
pip install huggingface_hub[cli]
huggingface-cli login
# Paste your hf_ token when prompted
```

### Option B: Environment Variable
1. Click 🔒 **Secrets** tab in Replit sidebar
2. Add secret:
   - Name: `HUGGINGFACE_TOKEN`
   - Value: `hf_your_token_here`

## Step 3: Request Llama 2 Access
1. Go to https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Click "Access repository" 
3. Fill form (name, organization, intended use)
4. Wait for approval (usually < 24 hours)

## Step 4: Test Download
```bash
python -c "
from transformers import AutoTokenizer
print('Testing Llama 2 access...')
try:
    AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    print('✅ Access granted!')
except:
    print('❌ Still waiting for approval')
"
```

## Alternative: Use Open Models (No Approval Required)
Your app now includes fallback models that work without approval:
- microsoft/DialoGPT-medium (conversation model)
- google/flan-t5-base (instruction following)
- These work immediately while you wait for Llama 2 approval