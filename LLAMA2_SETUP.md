# Llama 2 + LoRA Fine-tuning Setup for Course Assistant

## Why Llama 2 7B + LoRA is Recommended

**Perfect for your use case because:**
- **Memory retention**: Can learn from your course conversations
- **Course-specific knowledge**: Updates model weights with your Q&A patterns
- **Hardware efficient**: Works on RTX 3060 with 4-bit quantization
- **Privacy focused**: 100% local fine-tuning, no data leaves your machine
- **Conversation improvement**: Gets better at answering your specific course questions over time

## Setup Requirements

### Hardware Requirements
- **GPU**: RTX 3060 (8GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Storage**: Additional 15GB for model and training data

### Software Dependencies
```bash
pip install transformers>=4.35.0
pip install peft>=0.6.0
pip install bitsandbytes>=0.41.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
```

## Implementation Steps

### 1. Replace Current Model Loading
```python
# In local_models.py - replace Mistral with Llama 2 + LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

def load_llama2_with_lora():
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Low rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    return model
```

### 2. Conversation Learning System
```python
# Save conversations for fine-tuning
def save_conversation_for_training(question, answer, course_name):
    training_data = {
        "instruction": f"Answer this question about {course_name}:",
        "input": question,
        "output": answer,
        "course": course_name,
        "timestamp": datetime.now().isoformat()
    }
    
    # Append to training dataset
    with open(f"./training_data/{course_name}_conversations.jsonl", "a") as f:
        f.write(json.dumps(training_data) + "\n")

# Fine-tune model periodically
def fine_tune_on_conversations(course_name, num_conversations_threshold=10):
    # Load conversation data
    training_file = f"./training_data/{course_name}_conversations.jsonl"
    
    if not os.path.exists(training_file):
        return
        
    # Count conversations
    with open(training_file, 'r') as f:
        conversations = [json.loads(line) for line in f]
    
    if len(conversations) < num_conversations_threshold:
        return
        
    # Trigger fine-tuning
    fine_tune_model(conversations, course_name)
```

### 3. Hugging Face Authentication
You'll need a Hugging Face account and token to download Llama 2:

1. **Create account**: https://huggingface.co/
2. **Request Llama 2 access**: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
3. **Generate token**: Settings → Access Tokens → New Token
4. **Login locally**: `huggingface-cli login`

## Implementation Timeline

### Phase 1 (Day 1-2): Basic Setup
- Install dependencies
- Get Hugging Face access to Llama 2
- Test basic model loading

### Phase 2 (Day 3-5): Integration
- Replace Mistral with Llama 2 in your app
- Add conversation saving functionality
- Test basic Q&A without fine-tuning

### Phase 3 (Day 6-10): Learning System
- Implement LoRA fine-tuning pipeline
- Add automatic retraining after N conversations
- Test conversation memory improvement

## Expected Benefits

**After 20-50 conversations per course:**
- Better understanding of your specific terminology
- More relevant answers based on your question patterns  
- Improved context awareness for follow-up questions
- Course-specific knowledge that persists between sessions

**Storage Growth:**
- Base model: ~7GB (one-time)
- LoRA weights per course: ~50-100MB each
- Training conversations: ~1-5MB per course

## Alternative: Quick Test Setup

If you want to test learning capability immediately:
```python
# Use Alpaca 7B (already instruction-tuned)
model_name = "chavinlo/alpaca-native"
# Smaller, faster, but less capable than Llama 2
```

## Migration Path

1. **Keep current system working** - Llama 2 setup runs in parallel
2. **Test with one course** - Compare Mistral vs Llama 2 responses
3. **Gradual migration** - Switch courses one by one
4. **Full deployment** - Replace Mistral completely once satisfied

Would you like me to start implementing this setup, or do you have questions about any aspect of the fine-tuning approach?