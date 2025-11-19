## Face-Diet Sample GUI

A comprehensive Tkinter GUI for face analysis using **PyTorch** (works on Macs without AVX!) with InsightFace for detection and PyTorch-based attribute analysis.

### Features

1. **Face Detection**: Detect multiple faces with bounding boxes using InsightFace
2. **Attribute Analysis**: Analyze each face for age, gender, emotion, ethnicity (PyTorch-based)
3. **Pose Estimation**: Extract head pose angles (yaw, pitch, roll) from InsightFace
4. **Facial Landmarks**: Display 106-point color-coded landmarks grouped by facial region
5. **Interactive Tooltips**: Hover over detected faces to see all analyzed attributes
6. **Database Storage**: Save detection results and metadata to SQLite

### Setup

1. Create a conda environment with Python 3.9:

```bash
conda create -n face-diet python=3.9 -y
conda activate face-diet
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Important**: This setup uses **PyTorch instead of TensorFlow** to avoid AVX CPU instruction requirements. Works on older Intel Macs!

### Run GUI

```bash
conda activate face-diet
python gui.py
```

**Startup Time**: 15-25 seconds (InsightFace model loading)

### Usage

1. Click **Open Image** to load a photo
2. Click **Detect Faces** to find all faces (shows bounding boxes and pose)
3. Click **Analyze Attributes** to run PyTorch-based analysis (age, gender, emotion, ethnicity)
4. Hover over any face to see detailed information
5. Click **Landmarks** to overlay 106-point facial landmarks (color-coded by region)
6. Click **Save to DB** to persist results

### Files

- `gui.py`: Tkinter app UI with interactive tooltips
- `face_utils.py`: InsightFace wrapper for detection, pose, and landmarks
- `pytorch_attributes.py`: **PyTorch-based** attribute analysis (NO TensorFlow/AVX!)
- `db.py`: SQLite schema and helpers
- `compare_faces.py`: CLI example for computing cosine similarity

### Technical Notes

- **No AVX Required**: Uses PyTorch which works on older CPUs
- **No TensorFlow**: Avoids all TensorFlow/AVX compatibility issues
- InsightFace: Face detection, 106-point landmarks, head pose
- PyTorch: Face attribute analysis
- Tested on: macOS Intel x86_64 without AVX support

### Extending Attribute Analysis

The current `pytorch_attributes.py` provides demo data. To add real PyTorch models:

1. Age/Gender: Use pre-trained models from PyTorch Hub or Hugging Face
2. Emotion: Integrate PyTorch-based emotion recognition models
3. Ethnicity: Add fairness-aware classification models

All run on CPU without AVX requirements!

