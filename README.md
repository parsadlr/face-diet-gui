# Face-Diet: Video Face Analysis and Tracking

A comprehensive tool for analyzing faces in egocentric videos with face detection, identity tracking, and attribute extraction.

## Features

### Video Processing (New!)
- **Two-Pass Face Detection**: Collect all faces first, then assign consistent IDs via clustering
- **Face Identity Tracking**: Track same individuals across video frames using embedding similarity
- **Attribute Extraction**: Age, gender, race, emotion, head pose, and distance estimation
- **Smart Representative Selection**: For each identity, find the best instance (closest to cluster centroid)
- **Configurable Clustering**: Threshold-based or DBSCAN clustering methods
- **CSV & Video Output**: Results in CSV format + annotated video with face boxes and IDs

### Image Analysis (GUI)
- Face detection with bounding boxes
- Pose estimation (yaw, pitch, roll)
- 106-point facial landmarks with color coding
- Interactive tooltips showing face attributes
- SQLite database storage

## Setup

1. Create a conda environment with Python 3.9:

```bash
conda create -n face-diet python=3.9 -y
conda activate face-diet
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Video Processing (Command Line)

Process a video to detect and track faces:

```bash
python main.py path/to/video.mp4
```

This will create `video_faces.csv` with all detected face instances.

#### Options

**Output Control:**
```bash
python main.py video.mp4 -o results.csv              # Custom CSV output path
python main.py video.mp4 -v annotated.mp4            # Also create annotated video
```

**Sampling Rate:**
```bash
python main.py video.mp4 -s 30                       # Process every 30 frames
python main.py video.mp4 -s 1                        # Process every frame (slow!)
```

**Clustering Methods:**
```bash
# Threshold-based (simple, interpretable)
python main.py video.mp4 -c threshold -t 0.6         # Similarity threshold = 0.6

# DBSCAN (automatic, handles noise)
python main.py video.mp4 -c dbscan --dbscan-eps 0.4 --dbscan-min-samples 2
```

**Hardware:**
```bash
python main.py video.mp4 --gpu                       # Use GPU if available
```

**Full Example:**
```bash
python main.py my_video.mp4 \
  -o results/faces.csv \
  -v results/annotated_video.mp4 \
  -s 15 \
  -c threshold \
  -t 0.65 \
  --gpu
```

#### Output Format

**CSV Structure:**
Each row represents one face instance in a frame:
```
frame_number, time_seconds, face_id, bbox_x, bbox_y, bbox_w, bbox_h, 
pose_yaw, pose_pitch, pose_roll, age, gender, race, emotion, distance_estimate
```

Example:
```csv
frame_number,time_seconds,face_id,bbox_x,bbox_y,bbox_w,bbox_h,pose_yaw,pose_pitch,pose_roll,age,gender,race,emotion,distance_estimate
0,0.000,FACE_000,320,180,150,180,5.23,-2.15,1.03,28,Man,white,happy,1.000
30,1.000,FACE_000,325,185,145,175,8.45,-1.22,0.98,28,Man,white,neutral,1.034
30,1.000,FACE_001,580,220,120,140,15.32,3.45,-2.11,35,Woman,asian,neutral,1.250
```

**Annotated Video:**
- Bounding boxes around detected faces
- Face ID labels
- Attributes (age, gender, emotion) displayed as text overlay

### Image Analysis (GUI)

For static image analysis with interactive GUI:

```bash
python gui.py
```

**GUI Usage:**
1. Click **Open Image** to load a photo
2. Click **Detect Faces** to find all faces
3. Hover over faces to see attributes
4. Click **Landmarks** to show 106-point landmarks
5. Click **Save to DB** to persist results

## Project Structure

### Video Processing Modules
- `main.py` - Command-line entry point with configuration
- `video_processor.py` - Two-pass video processing orchestration
- `face_detection.py` - Face detection, embedding extraction, and clustering
- `face_attributes.py` - Attribute extraction (pose, age, gender, race, emotion, distance)
- `utils.py` - Helper functions (quality scoring, blur detection, CSV utilities)

### Image Analysis Modules (Legacy)
- `gui.py` - GUI for static image analysis
- `face_utils.py` - Face detection utilities
- `attribute_analysis.py` - DeepFace attribute analysis
- `db.py` - SQLite database operations
- `compare_faces.py` - Face similarity comparison

## Two-Pass Processing Pipeline

### Why Two-Pass?

Traditional single-pass tracking suffers from:
- **Drift**: Embeddings change slightly frame-by-frame, causing ID switches
- **Occlusions**: Face disappearing and reappearing gets new ID
- **Poor Representatives**: First detection might not be best quality

Two-pass approach solves these issues:

**Pass 1: Collection**
1. Process sampled frames
2. Detect all faces and extract embeddings
3. Extract attributes for each face instance
4. Store everything in memory

**Pass 2: Clustering & Assignment**
1. Cluster all embeddings using similarity/DBSCAN
2. Assign consistent face IDs based on clusters
3. Find representative instance per ID (closest to cluster centroid)
4. Write CSV and annotated video

### Clustering Methods

**Threshold-based** (default, `threshold=0.6`):
- Simple and interpretable
- Two faces with similarity > threshold = same person
- Good for high-quality videos with clear faces

**DBSCAN** (`eps=0.4, min_samples=2`):
- Automatic clustering
- Handles noise and outliers
- Good for varying lighting/angles

## Technical Details

### Face Detection
- **InsightFace**: ONNX-based face detector (buffalo_l model)
- 512-dimensional embeddings for identity matching
- 106-point landmarks + 3D head pose estimation

### Attribute Analysis
- **DeepFace**: Age, gender, race, emotion prediction
- Separate models for each attribute
- Runs on cropped face regions

### Distance Estimation
- Simple inverse relationship: `distance = reference_height / bbox_height`
- Reference: 150px face height = distance 1.0
- Larger distance value = face is farther away

### Performance
- **Sampling Rate**: For 30fps video, use `-s 30` to process 1 fps (good balance)
- **Memory**: ~2KB per face instance (embeddings + attributes)
- **Speed**: ~1-2 seconds per frame (CPU), ~0.3-0.5 seconds (GPU)

## Examples

### Process entire video at 1 fps sampling
```bash
python main.py meeting.mp4 -s 30 -o meeting_faces.csv
```

### High-quality annotation video
```bash
python main.py presentation.mp4 -s 15 -v annotated.mp4 --gpu
```

### Use DBSCAN for difficult conditions
```bash
python main.py outdoor.mp4 -c dbscan --dbscan-eps 0.35
```

### Process every frame (slow but thorough)
```bash
python main.py short_clip.mp4 -s 1
```

## Troubleshooting

**Error: "Cannot open video"**
- Check video file path and format (MP4, AVI supported)
- Try re-encoding: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

**Too many/few unique IDs detected**
- Adjust similarity threshold: `-t 0.5` (more IDs) or `-t 0.7` (fewer IDs)
- Try DBSCAN: `-c dbscan --dbscan-eps 0.35`

**Slow processing**
- Increase sampling rate: `-s 60` (process every 2 seconds)
- Use GPU: `--gpu`
- Skip video output (only CSV is much faster)

**AttributeError or import errors**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)

## Citation

This project uses:
- [InsightFace](https://github.com/deepinsight/insightface) for face detection
- [DeepFace](https://github.com/serengil/deepface) for attribute analysis
- [scikit-learn](https://scikit-learn.org/) for clustering

## License

See individual model licenses for InsightFace and DeepFace.
