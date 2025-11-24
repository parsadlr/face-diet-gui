# GPU Usage Analysis

## Current Status

### ✅ GPU IS Being Used

**Evidence from nvidia-smi:**
- **GPU Utilization**: 18% (active but not fully utilized)
- **GPU Memory**: 947MB / 8188MB (11.6% used)
- **Python Process**: Running on GPU (PID 3796)

**Evidence from Code:**
- All InsightFace models are using `CUDAExecutionProvider`
- Models loaded with CUDA:
  - `1k3d68.onnx` (3D landmarks) ✓
  - `2d106det.onnx` (2D landmarks) ✓
  - `det_10g.onnx` (face detection) ✓
  - `genderage.onnx` (age/gender) ✓
  - `w600k_r50.onnx` (recognition) ✓

## Why GPU Speedup May Be Limited

### 1. **DeepFace Runs on CPU** ⚠️ (Major Bottleneck)
- DeepFace uses TensorFlow, which runs on CPU by default
- Extracts: age, gender, race, emotion
- This is likely **slower than GPU face detection**
- Each face requires separate DeepFace call

### 2. **Video I/O Bottleneck**
- Reading frames from disk is CPU-bound
- Video decoding happens on CPU
- Can't be parallelized with GPU processing

### 3. **Single Frame Processing**
- Processing one frame at a time
- GPU works best with batch processing
- Small workloads don't fully utilize GPU

### 4. **GPU Utilization Pattern**
- 18% average utilization is typical for this workload
- GPU is active but waiting for CPU work
- Memory transfer overhead for small batches

## Performance Breakdown (Estimated)

For each frame with 1 face:
1. **Face Detection (GPU)**: ~50-100ms ✓ Fast
2. **Embedding Extraction (GPU)**: ~20-50ms ✓ Fast
3. **DeepFace Analysis (CPU)**: ~200-500ms ⚠️ **SLOW**
4. **Video I/O (CPU)**: ~10-30ms
5. **Other processing**: ~10-20ms

**Total per frame**: ~290-700ms
- GPU portion: ~70-150ms (20-25%)
- CPU portion: ~220-550ms (75-80%)

## Recommendations

### Option 1: Skip DeepFace (Fastest)
If you don't need age/gender/race/emotion:
- Use only InsightFace (all GPU)
- 3-5x faster overall
- Still get: detection, landmarks, pose, embeddings

### Option 2: Make DeepFace Optional
Add a flag to skip DeepFace:
```python
python main.py video.mp4 --gpu --skip-attributes
```

### Option 3: Batch Processing
Process multiple faces at once (requires code changes)

### Option 4: Use GPU for DeepFace
Configure TensorFlow to use GPU (complex setup)

## Verification Commands

```powershell
# Monitor GPU during processing
nvidia-smi -l 1

# Check which processes use GPU
nvidia-smi

# Test without DeepFace (if you modify code)
# Should see much higher GPU utilization
```

## Conclusion

**GPU is working correctly**, but:
- ✅ Face detection/recognition uses GPU
- ⚠️ DeepFace (attributes) uses CPU and is the bottleneck
- ⚠️ Overall speedup is limited by CPU-bound operations

**Expected speedup with --gpu**: 1.5-2x (not 3-5x)
- Without DeepFace: 3-5x speedup
- With DeepFace: 1.5-2x speedup

