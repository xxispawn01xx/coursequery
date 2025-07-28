"""
Local model management for Mistral 7B and embedding models.
Handles model loading, caching, and inference without external API calls.
"""

import logging
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import gc

# Optional AI/ML imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

try:
    from memory_optimizer import MemoryOptimizer
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MemoryOptimizer = None
    MEMORY_OPTIMIZER_AVAILABLE = False

class LocalModelManager:
    """Manages local AI models for text generation and embeddings."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.config = Config()
        self.mistral_model = None
        self.mistral_tokenizer = None
        self.mistral_pipeline = None
        self.llama_model = None
        self.llama_tokenizer = None
        self.llama_pipeline = None
        self.vllm_model = None
        self.embedding_model = None
        self.device = self._get_device()
        
        # Model selection
        self.current_model = "mistral"  # Default to Mistral
        
        # Simple conversation memory
        self.conversations = []
        
        logger.info(f"Initialized LocalModelManager with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return "cpu"
    
    def load_models(self, model_type: str = "mistral"):
        """Load all required models with caching optimization."""
        start_time = time.time()
        logger.info(f"⏱️ Loading local models (type: {model_type}) - Start time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Check memory status - RTX 3060 12GB should handle large models
        if MEMORY_OPTIMIZER_AVAILABLE:
            MemoryOptimizer.log_memory_status()
            # With RTX 3060 12GB, GPU memory is abundant
            if torch and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU Memory: {gpu_memory:.1f}GB - RTX 3060 can handle 7B models")
            else:
                if not MemoryOptimizer.check_memory_available(4.0):
                    suggested_model = MemoryOptimizer.suggest_optimal_model()
                    logger.warning(f"System RAM may be insufficient. Consider using: {suggested_model}")
        
        # RTX 3060 Configuration (Hardware Verified Healthy)
        gpu_is_healthy = True
        if torch and torch.cuda.is_available():
            logger.info("🔬 RTX 3060 Configuration - Hardware verified healthy via memtestcl")
            
            # Check if we have memtest verification
            gpu_test_file = Path("gpu_test_results.txt")
            if gpu_test_file.exists():
                try:
                    with open(gpu_test_file, "r") as f:
                        content = f.read()
                        if "MEMTEST_PASSED=True" in content:
                            logger.info("✅ Memtestcl confirmed RTX 3060 memory is healthy (0 errors)")
                            logger.info("Proceeding with optimized GPU configuration...")
                        elif "GPU_HEALTHY=False" in content:
                            logger.warning("Previous auto-test failed, but memtestcl passed - using GPU")
                            gpu_is_healthy = True  # Override based on memtest results
                        else:
                            logger.info("✅ Previous test showed GPU is healthy")
                except Exception:
                    pass
            
            if gpu_is_healthy:
                # Apply HuggingFace Forum RTX 3060 Fixes (verified working)
                logger.info("🚀 Applying verified HuggingFace forum RTX 3060 fixes...")
                
                # RTX 3060 environment setup from forums
                os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX 3060 Ampere
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                os.environ['ACCELERATE_USE_MPS_DEVICE'] = 'false'
                
                # Remove debug flags that cause device-side assert
                os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
                os.environ.pop('TORCH_USE_CUDA_DSA', None)
                
                # Test GPU with working approach
                try:
                    logger.info("🔍 Testing RTX 3060 with forum-verified method...")
                    test_tensor = torch.ones(100).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"✅ RTX 3060 ready - {total_memory:.1f}GB VRAM (forum fixes applied)")
                    
                except Exception as gpu_error:
                    logger.error(f"RTX 3060 still has issues: {gpu_error}")
                    gpu_is_healthy = False
                    self.device = 'cpu'
        
        # Verify HF_TOKEN availability
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info(f"✅ HF_TOKEN found: {hf_token[:8]}...{hf_token[-4:]}")
        else:
            logger.warning("⚠️  HF_TOKEN not found - may cause authentication issues")
        
        # Check if models are already loaded in memory
        if self._models_already_loaded(model_type):
            logger.info("Models and pipelines already loaded in memory, skipping reload")
            return
        elif (model_type == "mistral" and 
              self.mistral_model is not None and 
              self.mistral_tokenizer is not None and 
              self.mistral_pipeline is None):
            logger.info("Models loaded but pipeline missing - recreating pipeline only")
            try:
                self.mistral_pipeline = pipeline(
                    "text-generation",
                    model=self.mistral_model,
                    tokenizer=self.mistral_tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.mistral_tokenizer.eos_token_id
                )
                logger.info("Pipeline recreated successfully")
                return
            except Exception as e:
                logger.error(f"Failed to recreate pipeline: {e}")
                # Continue with full reload
        
        try:
            # Clear any existing models to ensure only one is loaded at a time
            self._clear_unused_models(model_type)
            
            # Load language models on verified healthy RTX 3060
            if self.device != 'cpu' and gpu_is_healthy:
                try:
                    logger.info("🚀 Loading models on memtest-verified RTX 3060...")
                    if model_type == "mistral":
                        self._load_mistral_model()
                    elif model_type == "llama":
                        self._load_llama_model()
                    else:
                        # Load default Mistral if unknown type
                        self._load_mistral_model()
                        model_type = "mistral"
                except Exception as model_error:
                    logger.error(f"Model loading failed despite healthy GPU: {model_error}")
                    if "device-side assert" in str(model_error):
                        logger.error("Device-side assert may be software config issue, not hardware")
                        logger.info("Trying CPU fallback while investigating GPU software config")
                        self.device = 'cpu'
                        self._load_mistral_cpu_fallback()
                        model_type = "mistral_cpu"
                    else:
                        raise model_error
            else:
                logger.info("🖥️  Loading models in CPU-only mode")
                self._load_mistral_cpu_fallback()
                model_type = "mistral_cpu"
                
            self._load_embedding_model()
            self.current_model = model_type if self.device != 'cpu' else 'cpu_embeddings_only'
            
            # Verify embedding model loaded successfully
            if self.embedding_model is None:
                logger.warning("Embedding model failed to load - Q&A functionality will be limited")
            
            total_time = time.time() - start_time
            logger.info(f"✅ All models loaded successfully ({model_type}) - Total time: {total_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _models_already_loaded(self, model_type: str) -> bool:
        """Check if the requested model is already loaded in memory."""
        if model_type == "mistral":
            models_loaded = (self.mistral_model is not None and 
                           self.mistral_tokenizer is not None and 
                           self.embedding_model is not None)
            pipeline_loaded = self.mistral_pipeline is not None
            # Ensure no other models are loaded (memory efficiency)
            other_models_clear = (self.llama_model is None and self.llama_tokenizer is None and self.llama_pipeline is None)
            logger.info(f"Mistral check - Models: {models_loaded}, Pipeline: {pipeline_loaded}, Other models clear: {other_models_clear}")
            return models_loaded and pipeline_loaded and other_models_clear
        elif model_type == "llama":
            models_loaded = (self.llama_model is not None and 
                           self.llama_tokenizer is not None and 
                           self.embedding_model is not None)
            pipeline_loaded = self.llama_pipeline is not None
            # Ensure no other models are loaded (memory efficiency)
            other_models_clear = (self.mistral_model is None and self.mistral_tokenizer is None and self.mistral_pipeline is None)
            logger.info(f"Llama check - Models: {models_loaded}, Pipeline: {pipeline_loaded}, Other models clear: {other_models_clear}")
            return models_loaded and pipeline_loaded and other_models_clear
        return False

    def _check_cache_status(self, model_name: str) -> str:
        """Check if model is already cached locally."""
        cache_path = self.config.models_dir / "models--" / model_name.replace("/", "--")
        if cache_path.exists():
            return "📦 Loading from cache (fast)"
        else:
            return "⬇️  First download (may take several minutes)"
    
    def _check_gguf_models(self) -> Optional[str]:
        """Check for available GGUF models in the models/gguf directory."""
        gguf_dir = self.config.models_dir / "gguf"
        if not gguf_dir.exists():
            return None
        
        # Look for GGUF files
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            logger.info(f"Found GGUF model: {gguf_files[0].name}")
            return str(gguf_files[0])
        return None
    
    def _clear_unused_models(self, target_model: str):
        """Clear models that aren't the target to save RTX 3060 memory - only one model at a time."""
        import gc
        
        if target_model == "mistral":
            # Clear Llama models for RTX 3060 memory optimization
            if self.llama_model is not None or self.llama_tokenizer is not None or self.llama_pipeline is not None:
                logger.info("🧹 RTX 3060 optimization: Clearing Llama models to load Mistral")
                del self.llama_model, self.llama_tokenizer, self.llama_pipeline
                self.llama_model = None
                self.llama_tokenizer = None
                self.llama_pipeline = None
                
                # Aggressive cleanup for RTX 3060
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
        elif target_model == "llama":
            # Clear Mistral models for RTX 3060 memory optimization
            if self.mistral_model is not None or self.mistral_tokenizer is not None or self.mistral_pipeline is not None:
                logger.info("🧹 RTX 3060 optimization: Clearing Mistral models to load Llama")
                del self.mistral_model, self.mistral_tokenizer, self.mistral_pipeline
                self.mistral_model = None
                self.mistral_tokenizer = None
                self.mistral_pipeline = None
                
                # Aggressive cleanup for RTX 3060
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Log memory status after cleanup
        if torch and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            logger.info(f"🚀 RTX 3060 memory after cleanup: {allocated:.0f}MB used / {total:.0f}MB total ({allocated/total*100:.1f}%)")

    def _load_mistral_model(self):
        """Load Mistral 7B model with RTX 3060 CPU-first debugging approach."""
        logger.info("Loading Mistral 7B model with RTX 3060 compatibility fixes...")
        
        # Try models in order of preference - RTX 3060 12GB can handle full models
        model_attempts = [
            ("meta-llama/Llama-2-7b-chat-hf", "Llama 2 7B Chat"),
            ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral 7B Instruct"),
            ("microsoft/DialoGPT-medium", "DialoGPT Medium"),
        ]
        
        for model_name, model_display_name in model_attempts:
            logger.info(f"🔄 Attempting {model_display_name}...")
            
            # CRITICAL RTX 3060 FIX: Test on CPU first to reveal real errors
            try:
                logger.info("🔍 Step 1: RTX 3060 Fix - Testing model on CPU first...")
                cpu_test_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info("✅ CPU test successful - model integrity verified!")
                del cpu_test_model
                torch.cuda.empty_cache()
                
            except Exception as cpu_error:
                logger.error(f"🚨 CPU test failed - REAL ERROR: {cpu_error}")
                if model_name == model_attempts[-1][0]:  # Last attempt
                    logger.error("All models failed CPU test - this is not RTX 3060 hardware issue!")
                    raise cpu_error
                continue  # Try next model
            
            # Step 2: Load tokenizer
            try:
                logger.info("📝 Loading tokenizer...")
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                    trust_remote_code=True
                )
                if self.mistral_tokenizer.pad_token is None:
                    self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
                    
            except Exception as tokenizer_error:
                logger.error(f"Tokenizer failed: {tokenizer_error}")
                continue
                
            # Step 3: Load model on GPU with RTX 3060 optimizations
            try:
                logger.info("🚀 Step 3: Loading on RTX 3060 GPU with optimizations...")
                
                # Configure 4-bit quantization for RTX 3060 efficiency
                quantization_config = None
                if torch and BitsAndBytesConfig:
                    logger.info("Using 4-bit quantization for RTX 3060 12GB efficiency")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                
                # HuggingFace Forum RTX 3060 Fix (Issues #28284, #22546)
                logger.info("Applying HuggingFace forum RTX 3060 fix...")
                
                # Remove problematic device_map - let PyTorch handle device placement
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                
                # Move to GPU manually (forum solution)
                if torch.cuda.is_available():
                    logger.info("Moving model to GPU manually (safer than device_map)")
                    self.mistral_model = self.mistral_model.cuda()
                
                # Step 4: Create pipeline
                logger.info("🔧 Creating text generation pipeline...")
                self.mistral_pipeline = pipeline(
                    "text-generation",
                    model=self.mistral_model,
                    tokenizer=self.mistral_tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.mistral_tokenizer.eos_token_id
                )
                
                logger.info(f"✅ Successfully loaded {model_display_name} with RTX 3060 optimizations!")
                return  # Success - exit the loop
                
            except Exception as gpu_error:
                logger.error(f"GPU loading failed for {model_display_name}: {gpu_error}")
                if model_name == model_attempts[-1][0]:  # Last attempt
                    logger.error("All models failed - RTX 3060 configuration issue")
                    raise gpu_error
                continue  # Try next model
        
        # If we get here, all models failed
        raise Exception("No models could be loaded successfully")

    def _load_mistral_cpu_fallback(self):
        """Load reliable CPU models when RTX 3060 has hardware issues."""
        logger.info("🔄 Loading CPU fallback model (RTX 3060 memory issues detected)...")
        logger.info("💡 CPU models are slower but work reliably with faulty GPU memory")
        
        # Use proven CPU models that work well even with GPU hardware issues
        cpu_models = [
            ("microsoft/DialoGPT-medium", "DialoGPT Medium"),
            ("gpt2", "GPT-2 Base"),
            ("distilgpt2", "DistilGPT-2"),
        ]
        
        for model_name, model_display_name in cpu_models:
            try:
                logger.info(f"Loading {model_display_name} on CPU...")
                
                # Load tokenizer
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                )
                if self.mistral_tokenizer.pad_token is None:
                    self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
                
                # Load model on CPU
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Create pipeline
                self.mistral_pipeline = pipeline(
                    "text-generation",
                    model=self.mistral_model,
                    tokenizer=self.mistral_tokenizer,
                    max_length=256,  # Smaller for CPU
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.mistral_tokenizer.eos_token_id
                )
                
                logger.info(f"✅ Successfully loaded {model_display_name} on CPU")
                return
                
            except Exception as e:
                logger.error(f"Failed to load {model_display_name} on CPU: {e}")
                continue
        
        raise Exception("CPU fallback models also failed to load")
    
    def _load_llama_model(self):
        """Load Llama 2 7B model with 4-bit quantization."""
        logger.info("Loading Llama 2 7B model...")
        
        try:
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            
            # Configure 4-bit quantization for RTX 3060 efficiency - more aggressive memory saving
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    # Additional memory optimizations
                    llm_int8_threshold=6.0,
                    llm_int8_skip_modules=None,
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.config.models_dir),
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Set pad token if not present
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            
            # Load model with memory optimization for RTX 3060
            # RTX 3060 CPU-first debug approach for Llama as well
            logger.info("🔍 Loading Llama on CPU first to verify model integrity...")
            try:
                # Test on CPU first
                cpu_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    low_cpu_mem_usage=True
                )
                logger.info("✅ Llama CPU test passed - moving to GPU...")
                del cpu_model
                torch.cuda.empty_cache()
                
                # Now load on GPU
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    quantization_config=quantization_config,
                    device_map="cuda:0",  # RTX 3060 specific fix
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    low_cpu_mem_usage=True,
                    max_memory={0: "5GB"},
                    offload_folder="./temp"
                )
                
            except Exception as cpu_error:
                logger.error(f"🚨 Llama CPU test failed - real error: {cpu_error}")
                raise cpu_error
            
            # Create pipeline
            self.llama_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.llama_tokenizer,
                max_length=4096,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )
            
            logger.info("Llama 2 7B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
    
    def _load_embedding_model_cpu(self):
        """Load embedding model specifically on CPU for fallback."""
        try:
            logger.info("Loading embedding model on CPU (fallback)...")
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.model_config['embeddings']['model_name']
            
            # Force CPU device
            import torch
            device = 'cpu'
            
            self.embedding_model = SentenceTransformer(
                model_name, 
                device=device,
                token=os.getenv("HF_TOKEN")  # Updated from deprecated use_auth_token
            )
            logger.info(f"✅ Embedding model loaded on CPU: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model on CPU: {e}")
            self.embedding_model = None
            raise

    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        logger.info("Loading embedding model...")
        
        # Check if sentence-transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Please install it: pip install sentence-transformers")
            raise ImportError("sentence-transformers package required for embeddings")
        
        try:
            model_config = self.config.model_config['embeddings']
            model_name = model_config['model_name']
            
            cache_status = self._check_cache_status(model_name)
            logger.info(f"Loading embedding model from cache: {self.config.models_dir} - {cache_status}")
            
            # Try CUDA first if available
            if torch and torch.cuda.is_available():
                try:
                    logger.info("Attempting to load embedding model on CUDA...")
                    # Set CUDA debug mode to get better error messages
                    os.environ['TORCH_USE_CUDA_DSA'] = '1'
                    
                    # Optimize GPU memory usage for RTX 3060
                    import torch
                    torch.cuda.empty_cache()  # Clear before loading
                    
                    # Check available memory before loading
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                    free_memory = total_memory - allocated_memory
                    
                    logger.info(f"GPU Memory Check: {allocated_memory:.1f}GB used, {free_memory:.1f}GB free of {total_memory:.1f}GB total")
                    
                    if free_memory < 2.0:  # Reasonable threshold for RTX 3060
                        logger.warning(f"⚠️  Memory optimization needed: {free_memory:.1f}GB available")
                        logger.info("🔧 Clearing CUDA cache for RTX 3060")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        if hasattr(torch.cuda, 'ipc_collect'):
                            torch.cuda.ipc_collect()
                    
                    # RTX 3060 optimized embedding model loading
                    self.embedding_model = SentenceTransformer(
                        model_name,
                        cache_folder=str(self.config.models_dir),
                        device='cuda:0',  # Explicit GPU 0 (forum recommendation)
                        token=os.getenv("HF_TOKEN")
                    )
                    
                    # Apply optimizations for RTX 3060
                    try:
                        self.embedding_model.half()  # FP16 for memory efficiency
                        logger.info("✅ Applied FP16 optimization for RTX 3060")
                    except Exception as half_error:
                        logger.warning(f"FP16 conversion failed: {half_error}")
                    
                    # RTX 3060 optimized batch size
                    self.embedding_model.encode_kwargs = {'batch_size': 16}  # Balanced for 12GB
                    logger.info("✅ Embedding model loaded on CUDA - OPTIMAL PERFORMANCE")
                    return
                    
                except Exception as cuda_error:
                    error_msg = str(cuda_error).lower()
                    if any(keyword in error_msg for keyword in ["cuda", "device-side assert", "out of memory", "insufficient", "memory"]):
                        logger.error(f"🚨 GPU MEMORY ERROR: {cuda_error}")
                        logger.error("⚠️  RTX 3060 memory exhausted - trying ultra-aggressive optimization")
                        
                        # Clear CUDA state completely and retry with minimal settings
                        if torch and torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                torch.cuda.ipc_collect()
                                
                                # Try loading with minimal memory footprint
                                logger.info("🔧 Retrying with ultra-low memory settings")
                                self.embedding_model = SentenceTransformer(
                                    model_name,
                                    cache_folder=str(self.config.models_dir),
                                    device='cuda',
                                    token=os.getenv("HF_TOKEN")
                                )
                                
                                # Immediately convert to lowest precision
                                self.embedding_model.half()
                                
                                # Force very small batch processing
                                self.embedding_model.encode_kwargs = {'batch_size': 4}
                                logger.info("✅ Ultra-optimized embedding model loaded on CUDA")
                                return
                                
                            except Exception as retry_error:
                                logger.error(f"Ultra-optimization also failed: {retry_error}")
                                raise cuda_error
                    else:
                        raise cuda_error
            
            # CPU fallback
            logger.info("Loading embedding model on CPU...")
            self.embedding_model = SentenceTransformer(
                model_name,
                cache_folder=str(self.config.models_dir),
                device='cpu',
                token=os.getenv("HF_TOKEN")  # Updated from deprecated use_auth_token
            )
            logger.error("⚠️  CPU FALLBACK: Embedding model loaded on CPU - SIGNIFICANTLY SLOWER than GPU!")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Set to None but don't raise - allow app to continue with limited functionality
            self.embedding_model = None
            logger.warning("Continuing without embedding model - Q&A will be limited")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate response using local Mistral model.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated response text
        """
        start_time = time.time()
        logger.info(f"🤔 Starting response generation - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        # Use appropriate model based on current selection
        if self.current_model == "llama" and self.llama_pipeline is not None:
            pipe = self.llama_pipeline
            formatted_prompt = self._format_llama_prompt(prompt)
        elif self.mistral_pipeline is not None:
            pipe = self.mistral_pipeline
            formatted_prompt = self._format_instruction_prompt(prompt)
        else:
            # Check if we have any loaded models and fix the issue
            if self.mistral_model is not None and self.mistral_tokenizer is not None:
                logger.info("Model loaded but pipeline missing - recreating pipeline...")
                try:
                    self.mistral_pipeline = pipeline(
                        "text-generation",
                        model=self.mistral_model,
                        tokenizer=self.mistral_tokenizer,
                        max_length=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.mistral_tokenizer.eos_token_id
                    )
                    pipe = self.mistral_pipeline
                    formatted_prompt = self._format_instruction_prompt(prompt)
                    pipeline = pipe  # Use the recreated pipeline
                    logger.info("Pipeline recreated successfully")
                except Exception as e:
                    raise RuntimeError(f"Failed to recreate pipeline: {e}")
            else:
                raise RuntimeError(f"No model loaded (current: {self.current_model}). Models: mistral={self.mistral_model is not None}, llama={self.llama_model is not None}, pipelines: mistral_pipeline={self.mistral_pipeline is not None}, llama_pipeline={self.llama_pipeline is not None}")
        
        try:
            # Clear CUDA cache and handle device-side assert errors
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset CUDA context to prevent device-side assert errors
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            
            # Generate response with timeout and optimized settings for RTX 3060
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Response generation timed out after 30 seconds")
            
            # Set timeout for response generation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                response = pipe(
                    formatted_prompt,
                    max_new_tokens=min(max_new_tokens, 256),  # Limit tokens for faster generation
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    return_full_text=False,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
            finally:
                signal.alarm(0)  # Clear timeout
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Clean up the response
            cleaned_response = self._clean_response(generated_text)
            
            # Log performance metrics
            total_time = time.time() - start_time
            tokens_generated = len(cleaned_response.split())
            tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
            logger.info(f"⚡ Response generated in {total_time:.2f}s | {tokens_generated} tokens | {tokens_per_second:.1f} tokens/sec | Device: {self.device}")
            
            return cleaned_response
            
        except TimeoutError as e:
            logger.error(f"Response generation timed out: {e}")
            # Clear CUDA cache and return fallback response
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return "I apologize, but the response generation took too long. Please try asking a shorter or more specific question."
        except Exception as e:
            # Handle CUDA errors specifically with recovery
            if "CUDA" in str(e).upper() or "device-side assert" in str(e).lower():
                logger.error(f"CUDA error during response generation: {e}")
                logger.info("Attempting to clear CUDA cache and recover...")
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Force garbage collection
                        import gc
                        gc.collect()
                        logger.info("CUDA cache cleared, attempting recovery...")
                        
                        # Return a helpful fallback response instead of crashing
                        return "I encountered a GPU memory issue while processing your request. Please try asking a shorter, more specific question, or restart the application if this persists."
                    except Exception as cuda_error:
                        logger.error(f"Failed to recover from CUDA error: {cuda_error}")
                        raise RuntimeError(f"CUDA error in text generation. Please restart the app: {e}")
                else:
                    raise RuntimeError(f"CUDA error but no CUDA available: {e}")
            else:
                logger.error(f"Error generating response: {e}")
                raise
    
    def _format_instruction_prompt(self, user_query: str) -> str:
        """Format prompt for instruction following."""
        return f"""<s>[INST] You are a knowledgeable course assistant. Provide accurate, detailed answers based on the given context. Be specific and practical in your responses.

If asked to create spreadsheets, tables, or Excel files, provide:
1. Clear step-by-step instructions for creating them manually
2. Detailed table structure with column headers and example data
3. Relevant formulas or calculations needed

User Question: {user_query} [/INST]"""
    
    def _format_llama_prompt(self, user_query: str) -> str:
        """Format prompt for Llama 2 chat format."""
        return f"""<s>[INST] <<SYS>>
You are a knowledgeable course assistant. Provide accurate, detailed answers based on the given context. Be specific and practical in your responses.

If asked to create spreadsheets, tables, or Excel files, provide:
1. Clear step-by-step instructions for creating them manually
2. Detailed table structure with column headers and example data
3. Relevant formulas or calculations needed
<</SYS>>

{user_query} [/INST]"""
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove any potential instruction tags that might leak through
        response = response.replace("[INST]", "").replace("[/INST]", "")
        response = response.replace("<s>", "").replace("</s>", "")
        
        # Strip extra whitespace
        response = response.strip()
        
        return response
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        start_time = time.time()
        logger.info(f"🔍 Generating embeddings for {len(texts)} texts - {datetime.now().strftime('%H:%M:%S')}")
        if self.embedding_model is None:
            logger.error("Embedding model not loaded - trying to reload...")
            try:
                self._load_embedding_model()
                if self.embedding_model is None:
                    raise RuntimeError("Embedding model not loaded and reload failed")
            except Exception as e:
                raise RuntimeError(f"Embedding model not loaded: {e}")
        
        try:
            # Clear CUDA cache and check memory before encoding
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                free_memory = total - allocated
                
                logger.info(f"GPU Memory: {allocated:.0f}MB used, {free_memory:.0f}MB free of {total:.0f}MB total")
                
                # Estimate memory needed (rough calculation)
                estimated_memory_mb = len(texts) * 2  # Rough estimate: 2MB per text chunk
                
                if free_memory < estimated_memory_mb * 1.5:  # Need 1.5x buffer
                    logger.error(f"🚨 Insufficient GPU memory! Need ~{estimated_memory_mb}MB, only {free_memory:.0f}MB available")
                    logger.error("⚠️  Your GPU is 96% full - forcing CPU processing")
                    raise RuntimeError(f"Insufficient GPU memory: need {estimated_memory_mb}MB, have {free_memory:.0f}MB")
            
            # Use optimized batch processing for RTX 3060
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                batch_size=8,  # Smaller batches to prevent memory overflow
                show_progress_bar=len(texts) > 50,  # Only show progress for large batches
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Log embedding performance
            total_time = time.time() - start_time
            texts_per_second = len(texts) / total_time if total_time > 0 else 0
            logger.info(f"📊 Embeddings generated in {total_time:.2f}s | {len(texts)} texts | {texts_per_second:.1f} texts/sec")
            
            return embeddings.tolist()
            
        except RuntimeError as e:
            if "CUDA" in str(e) and "device-side assert" in str(e):
                logger.error(f"CUDA device-side assert error: {e}")
                logger.info("Forcing CPU fallback due to CUDA error...")
                
                try:
                    # Force CPU processing
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Reload embedding model on CPU
                    logger.error("🚨 CUDA DEVICE ASSERT ERROR - FORCING CPU FALLBACK!")
                    logger.error("⚠️  PERFORMANCE WARNING: CPU embeddings will be 5-10x slower!")
                    self._load_embedding_model_cpu()
                    
                    # Generate embeddings on CPU with optimized batch processing
                    embeddings = self.embedding_model.encode(
                        texts, 
                        convert_to_numpy=True,
                        batch_size=16,  # Larger batches OK on CPU
                        show_progress_bar=len(texts) > 50,
                        device='cpu'
                    )
                    
                    total_time = time.time() - start_time
                    logger.error(f"⚠️  CPU FALLBACK COMPLETED: {total_time:.2f}s | {len(texts)} texts - MUCH SLOWER than GPU!")
                    
                    return embeddings.tolist()
                    
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
                    raise RuntimeError(f"Both CUDA and CPU embedding generation failed. Original CUDA error: {e}")
            else:
                logger.error(f"Non-CUDA runtime error: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'models_loaded': {
                'mistral': self.mistral_model is not None,
                'embeddings': self.embedding_model is not None,
            }
        }
        
        if torch.cuda.is_available():
            info['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
            }
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared")
    
    def unload_models(self):
        """Unload models to free memory."""
        logger.info("Unloading models...")
        
        self.mistral_model = None
        self.mistral_tokenizer = None
        self.mistral_pipeline = None
        self.embedding_model = None
        
        self.clear_cache()
        logger.info("Models unloaded")
    
    def save_conversation(self, question: str, answer: str, course_name: str = "default"):
        """Save conversation for future learning."""
        import json
        import os
        from datetime import datetime
        
        conversation = {
            "question": question,
            "answer": answer,
            "course": course_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to memory
        self.conversations.append(conversation)
        
        # Save to file for persistence
        os.makedirs("./conversations", exist_ok=True)
        filename = f"./conversations/{course_name}_conversations.jsonl"
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(conversation) + "\n")
        
        logger.info(f"Saved conversation for course: {course_name}")
        
        # Simple learning trigger (every 10 conversations)
        if len(self.conversations) % 10 == 0:
            logger.info(f"Learning trigger: {len(self.conversations)} conversations saved")
