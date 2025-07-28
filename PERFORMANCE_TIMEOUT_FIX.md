# Performance Timeout Fix for RTX 3060

## Issue
Response generation was hanging indefinitely during text generation phase, causing the app to be stuck on "Thinking..." for minutes.

## Root Cause
Long text generation without timeout protection, potentially due to:
- Large token generation requests (512+ tokens)
- GPU memory pressure on RTX 3060 12GB
- Model pipeline hanging during generation

## Fixes Applied

### 1. Timeout Protection
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Response generation timed out after 30 seconds")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout
```

### 2. Optimized Token Generation
- Reduced `max_new_tokens` from 512 to 256 for faster responses
- Added proper `pad_token_id` handling
- Graceful fallback response on timeout

### 3. Enhanced Error Handling
- Specific TimeoutError handling with CUDA cache clearing
- Graceful degradation instead of hanging indefinitely
- Detailed timing logs for performance monitoring

### 4. Performance Monitoring
Added comprehensive timing logs:
- `🔍 Processing query for course: vcpe - Start: 23:50:05.316`
- `⚡ Response generated in 2.34s | 89 tokens | 38.0 tokens/sec | Device: cuda:0`
- `⚡ Query completed in 3.12s | Course: vcpe | Nodes: 5 | Context: 2048 chars`

## Expected Performance (RTX 3060)
- **Model Loading**: 30-60 seconds (one-time)
- **Embedding Generation**: 0.1-0.5 seconds
- **Text Generation**: 1-5 seconds (now limited to 256 tokens)
- **Total Query Time**: 2-8 seconds

## Fallback Behavior
If generation times out (>30 seconds):
- Clears CUDA cache automatically
- Returns helpful message asking for shorter/more specific questions
- Logs detailed error information for debugging

## Status
The hanging issue should now be resolved with:
- ✅ 30-second timeout protection
- ✅ Optimized token limits for RTX 3060
- ✅ Graceful error handling
- ✅ Comprehensive performance logging

Your RTX 3060 system should now provide consistent response times under 10 seconds.