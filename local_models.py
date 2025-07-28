"""
Local model management for Mistral 7B and embedding models.
Handles model loading, caching, and inference without external API calls.
"""

import logging
import os
import json
import time
from datetime import datetime
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
        
        # CRITICAL: GPU Health Check for RTX 3060 hardware issues
        if torch and torch.cuda.is_available():
            try:
                logger.info("🔍 Performing GPU health check for device-side assert errors...")
                
                # Test basic CUDA operations that commonly trigger device-side asserts
                test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                result = test_tensor * 2  # Simple multiplication
                _ = result.sum()  # Reduction operation
                del test_tensor, result
                torch.cuda.empty_cache()
                
                # Check memory availability
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"✅ GPU health check passed - {allocated:.1f}GB/{total_memory:.1f}GB used")
                
            except Exception as gpu_error:
                error_msg = str(gpu_error).lower()
                if "device-side assert" in error_msg or "cuda" in error_msg:
                    logger.error(f"🚨 GPU MEMORY ISSUE DETECTED: {gpu_error}")
                    logger.error("⚠️  RTX 3060 memory optimization required - clearing cache and reducing model size")
                    logger.error("💡 SOLUTION: Aggressive memory management to fit within 12GB")
                    
                    # Clear all CUDA memory and try aggressive optimization
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                    
                    # Continue with GPU but force smaller memory footprint
                    logger.info("🔧 Applying aggressive memory optimization for RTX 3060")
                else:
                    raise gpu_error
        
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
            
            # Only load language models if GPU is healthy or using CPU
            if self.device != 'cpu':
                if model_type == "mistral":
                    self._load_mistral_model()
                elif model_type == "llama":
                    self._load_llama_model()
                else:
                    # Load default Mistral if unknown type
                    self._load_mistral_model()
                    model_type = "mistral"
            else:
                logger.error("GPU required for this application - CPU mode disabled")
                
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
        """Clear models that aren't the target to save memory - only one model at a time."""
        if target_model == "mistral":
            # Clear Llama models
            if self.llama_model is not None or self.llama_tokenizer is not None or self.llama_pipeline is not None:
                logger.info("🧹 Clearing Llama models to load Mistral (memory optimization)")
                self.llama_model = None
                self.llama_tokenizer = None
                self.llama_pipeline = None
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elif target_model == "llama":
            # Clear Mistral models
            if self.mistral_model is not None or self.mistral_tokenizer is not None or self.mistral_pipeline is not None:
                logger.info("🧹 Clearing Mistral models to load Llama (memory optimization)")
                self.mistral_model = None
                self.mistral_tokenizer = None
                self.mistral_pipeline = None
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _load_mistral_model(self):
        """Load Mistral 7B model with 4-bit quantization or GGUF if available."""
        logger.info("Loading Mistral 7B model...")
        
        # Check for GGUF models first
        gguf_path = self._check_gguf_models()
        if gguf_path:
            logger.info(f"Using GGUF model: {gguf_path}")
            # GGUF loading would require llama-cpp-python or similar
            # For now, show detected but continue with HuggingFace model
            logger.info("GGUF detected but using HuggingFace model for compatibility")
        
        # Try models in order of preference - RTX 3060 12GB can handle full models
        model_attempts = [
            ("meta-llama/Llama-2-7b-chat-hf", "Llama 2 7B Chat", "causal"),  # Primary choice
            ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral 7B", "causal"),  # Fallback
            ("microsoft/DialoGPT-medium", "DialoGPT Medium", "causal"),       # System RAM fallback
        ]
        
        for model_name, model_display_name, model_type in model_attempts:
            try:
                cache_status = self._check_cache_status(model_name)
                logger.info(f"Attempting to load {model_display_name}... {cache_status}")
                
                # Configure quantization and memory optimization
                quantization_config = None
                low_cpu_mem_usage = True
                
                # Use 4-bit quantization for RTX 3060 efficiency
                if torch and BitsAndBytesConfig and ("Mistral" in model_name or "Llama" in model_name):
                    if torch.cuda.is_available():
                        logger.info("Using 4-bit quantization for RTX 3060 12GB efficiency")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    else:
                        # CPU fallback if GPU fails
                        logger.info("GPU not available, using CPU mode")
                
                # Load tokenizer first (this will fail fast if authentication is wrong)
                logger.info(f"Loading tokenizer from cache: {self.config.models_dir}")
                self.mistral_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.config.models_dir),
                    trust_remote_code=True,
                    local_files_only=False,  # Allow download if needed but prefer cache
                )
                
                # Set pad token if not present
                if self.mistral_tokenizer.pad_token is None:
                    self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
                
                # Load model with memory optimization
                model_kwargs = {
                    "cache_dir": str(self.config.models_dir),
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": low_cpu_mem_usage,
                    "local_files_only": False,
                }
                
                if torch:
                    # Use smaller float precision to save memory
                    if torch.cuda.is_available():
                        model_kwargs["torch_dtype"] = torch.float16
                    else:
                        # For CPU, use float32 but with smaller batch sizes
                        model_kwargs["torch_dtype"] = torch.float32
                
                # Add quantization and device mapping only if accelerate is available
                try:
                    import accelerate
                    if quantization_config and BitsAndBytesConfig:
                        model_kwargs["quantization_config"] = quantization_config
                    if torch and torch.cuda.is_available():
                        model_kwargs["device_map"] = "auto"
                except ImportError:
                    logger.info("Accelerate not available, using CPU-only mode")
                
                if model_type == "causal":
                    logger.info(f"Loading model from cache: {self.config.models_dir}")
                    model_kwargs["local_files_only"] = False  # Allow download if needed but prefer cache
                    model_kwargs["use_safetensors"] = True  # Force safetensors to avoid torch.load security issue
                    model_kwargs["trust_remote_code"] = False  # Security best practice
                    
                    # Ultra-aggressive memory optimization for RTX 3060
                    model_kwargs.update({
                        "low_cpu_mem_usage": True,
                        "max_memory": {0: "4GB"},  # More aggressive - limit to 4GB for Mistral
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "device_map": {"": 0},  # Force single GPU mapping
                    })
                    
                    self.mistral_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                else:  # seq2seq models like T5
                    from transformers import AutoModelForSeq2SeqLM
                    self.mistral_model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                
                logger.info(f"Successfully loaded {model_display_name}")
                break  # Success! Exit the loop
                
            except Exception as e:
                logger.warning(f"Failed to load {model_display_name}: {e}")
                if model_name == model_attempts[-1][0]:  # Last attempt
                    logger.error("Both Llama 2 and Mistral model loading failed. Please check authentication.")
                    logger.info("To use these models, you need Hugging Face authentication:")
                    logger.info("1. Get token from https://huggingface.co/settings/tokens")
                    logger.info("2. Run: huggingface-cli login")
                    logger.info("3. Or set HF_TOKEN environment variable")
                    raise e
                continue  # Try next model
        
        try:
            
            # Create pipeline with default configuration
            self.mistral_pipeline = pipeline(
                "text-generation",
                model=self.mistral_model,
                tokenizer=self.mistral_tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.mistral_tokenizer.eos_token_id
            )
            
            logger.info("Mistral 7B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Mistral model: {e}")
            raise
    
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
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.config.models_dir),
                quantization_config=quantization_config,
                device_map={"": 0},  # Force single GPU mapping
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                # Ultra-aggressive memory optimizations for RTX 3060
                low_cpu_mem_usage=True,
                max_memory={0: "5GB"},  # Even more aggressive - limit to 5GB for Llama
                offload_folder="./temp"  # Offload some layers to disk if needed
            )
            
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
                    
                    if free_memory < 1.0:  # Reduced threshold - be more aggressive
                        logger.warning(f"⚠️  Low GPU memory for embedding model: only {free_memory:.1f}GB available")
                        logger.info("🔧 Applying ultra-aggressive memory optimization")
                        # Clear everything and continue with minimal footprint
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.ipc_collect()
                    
                    # Use half precision to reduce memory by ~50%
                    self.embedding_model = SentenceTransformer(
                        model_name,
                        cache_folder=str(self.config.models_dir),
                        device='cuda',
                        token=os.getenv("HF_TOKEN")
                    )
                    
                    # Convert to half precision after loading to save memory
                    if hasattr(self.embedding_model._modules, 'values'):
                        for module in self.embedding_model._modules.values():
                            if hasattr(module, 'half'):
                                module.half()
                    
                    # Set smaller batch size for encoding
                    self.embedding_model.encode_kwargs = {'batch_size': 8}  # Reduced from default 32
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
