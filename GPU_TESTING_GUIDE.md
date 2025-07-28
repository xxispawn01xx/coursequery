# GPU Testing Guide for Used RTX 3060

## Official NVIDIA Testing Tools

### 1. NVIDIA GPU Memory Test (Recommended)
**Best for detecting memory issues like yours**
- Download: Search "NVIDIA GPU Memory Test" or "MemTestCL"
- Tests: Memory corruption, stability, temperature
- Duration: 15-30 minutes for thorough test
- **This is ideal for your device-side assert errors**

### 2. NVIDIA System Management Interface (nvidia-smi)
**Built into NVIDIA drivers**
```cmd
nvidia-smi
nvidia-smi -l 1  # Live monitoring
nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total --format=csv -l 1
```

### 3. NVIDIA Control Panel Stress Test
- Open NVIDIA Control Panel
- Go to "Help" → "System Information" 
- Run built-in diagnostics

### 4. MSI Afterburner + Kombustor
**Popular for stress testing**
- Download: MSI Afterburner + MSI Kombustor
- Tests: GPU stability, temperature, memory
- Good for sustained load testing

### 5. GPU-Z
**For detailed hardware information**
- Download: TechPowerUp GPU-Z
- Shows: Memory type, temperatures, clocks
- Sensor monitoring for real-time data

## Quick Command Line Tests

### Test 1: Basic CUDA Availability
```cmd
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Test 2: Memory Allocation Test
```cmd
python gpu_memory_test.py
```

### Test 3: NVIDIA Driver Test
```cmd
nvidia-smi --query-gpu=memory.total,memory.used,temperature.gpu --format=csv
```

## What Your Error Suggests

Your specific error: **"CUDA error: device-side assert triggered"**

**Likely causes in used GPUs:**
1. **Memory corruption** - Bad VRAM cells
2. **Overheating** - Thermal throttling causing instability  
3. **Power delivery issues** - Unstable voltage to GPU
4. **Driver conflicts** - Outdated or corrupted drivers

## Immediate Actions

### 1. Check Temperatures
```cmd
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1
```
**Normal:** Under 80°C under load
**Concerning:** Above 85°C

### 2. Update Drivers
- Download latest RTX 3060 drivers from NVIDIA
- Use DDU (Display Driver Uninstaller) for clean install

### 3. Memory Test Priority
**Download and run NVIDIA GPU Memory Test first** - this directly tests for the memory corruption that causes your specific error.

## System Commands to Run Now

```cmd
# Check current driver version
nvidia-smi

# Check for Windows memory issues  
sfc /scannow

# Check system memory
mdsched.exe
```

## Red Flags for Used GPU Issues

- **Temperature spikes** above 85°C
- **Inconsistent benchmark scores** 
- **Artifacts** in graphics
- **Random crashes** in different applications
- **Your current error** - device-side asserts

## Next Steps

1. **Run nvidia-smi** to check current status
2. **Download NVIDIA GPU Memory Test** for thorough memory testing
3. **Monitor temperatures** during embedding generation
4. **Test with minimal workload** to isolate the issue

The fact that your embedding model loads successfully but fails during computation strongly suggests **VRAM memory corruption** - exactly what NVIDIA GPU Memory Test is designed to detect.