# DeepFace Batch Processing Implementation

## Overview

DeepFace batch processing has been implemented to significantly speed up attribute extraction when processing multiple faces per frame. Instead of calling DeepFace individually for each face, we now process multiple faces in a single batch call.

## How It Works

### Before (Sequential)
- Process face 1 → DeepFace call 1 (load models, process, unload)
- Process face 2 → DeepFace call 2 (load models, process, unload)
- Process face 3 → DeepFace call 3 (load models, process, unload)
- **Total**: 3 model loads, 3 processing calls

### After (Batch)
- Process faces 1, 2, 3 → Single DeepFace batch call (load models once, process all, unload)
- **Total**: 1 model load, 1 processing call

## Performance Benefits

Expected improvements:
- **2-4x faster** for frames with multiple faces (2-8 faces)
- **Reduced model loading overhead** - models loaded once per batch instead of per face
- **Better GPU utilization** - batch processing is more efficient on GPU
- **Lower memory fragmentation** - fewer model load/unload cycles

## Usage

### Default (Batch Enabled)
```bash
python main.py video.mp4 --gpu
```
- Batch processing: **Enabled** (default batch size: 8)
- Automatically uses batch processing when 2+ faces detected

### Custom Batch Size
```bash
python main.py video.mp4 --gpu --batch-size 16
```
- Increase batch size for more faces per frame (uses more GPU memory)
- Decrease batch size (e.g., `--batch-size 4`) if GPU memory is limited

### Disable Batch Processing
```bash
python main.py video.mp4 --gpu --no-batch
```
- Falls back to parallel processing (ThreadPoolExecutor)
- Useful if batch processing causes issues

### With Profiling
```bash
python main.py video.mp4 --gpu --profile
```
- Shows `deepface_batch_analysis` timing in profiling output
- Compare with `deepface_analysis` to see improvement

## Implementation Details

### New Functions

1. **`extract_age_gender_race_emotion_batch()`**
   - Processes multiple faces in a single DeepFace call
   - Handles batching automatically (splits into batches of `batch_size`)
   - Falls back to individual processing if batch fails

2. **`extract_all_attributes_batch()`**
   - Extracts all attributes (pose + demographics) for multiple faces
   - Uses batch processing for DeepFace, individual processing for pose

### Processing Flow

1. **Frame with multiple faces detected**
2. **Extract poses individually** (must be done per-face)
3. **Batch process DeepFace** (all faces in one call)
4. **Combine results** (pose + demographics per face)

### Fallback Behavior

- If batch processing fails → falls back to individual calls
- If single face → uses individual processing (no overhead)
- If `--no-batch` flag → uses parallel processing instead

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size N` | 8 | Maximum faces per DeepFace batch |
| `--no-batch` | False | Disable batch processing, use parallel instead |
| `--max-workers N` | 4 | Parallel workers (used if batch disabled) |

## Expected Profiling Results

### Before (Sequential)
```
deepface_analysis: 116.96s (24.8% of total)
  - 1875 calls, 0.0624s per call
```

### After (Batch)
```
deepface_batch_analysis: ~30-50s (estimated 6-10% of total)
  - ~235 batch calls (1875 faces / 8 batch size)
  - ~0.13-0.21s per batch (8 faces)
  - ~0.016-0.026s per face (4x faster!)
```

## Tuning Recommendations

1. **Many faces per frame (5-10+)**: Increase `--batch-size` to 12-16
2. **Few faces per frame (1-3)**: Keep default (8) or reduce to 4
3. **GPU memory issues**: Reduce `--batch-size` to 4-6
4. **Single face frames**: No impact (uses individual processing)

## Troubleshooting

### Batch Processing Fails
- Check DeepFace version compatibility
- Try reducing `--batch-size`
- Use `--no-batch` to fall back to parallel processing

### Out of Memory Errors
- Reduce `--batch-size` (e.g., `--batch-size 4`)
- Ensure GPU has enough memory
- Close other GPU applications

### No Performance Improvement
- Check profiling output - verify `deepface_batch_analysis` is being used
- Ensure multiple faces per frame (batch only helps with 2+ faces)
- Check if models are being cached properly

