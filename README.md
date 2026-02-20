# Face-Diet

A GUI for processing egocentric video recordings to extract, review, cluster, and manually verify face instances across participants and sessions.

## Overview

The pipeline has four stages, each with its own GUI tab:

| Tab | Stage | What it does |
|-----|-------|--------------|
| 1 | Face Detection | Runs Stages 1 & 2 (InsightFace detection + DeepFace attribute extraction) on selected sessions. Writes `face_detections.csv` per session. |
| 2 | Face Instance Review | Reviewer marks individual detections as valid face / non-face. |
| 3 | Face ID Clustering | Runs graph-based community detection (Leiden/Louvain) across all sessions of a participant to assign global face IDs. |
| 4 | Face ID Review | Reviewer merges face IDs that were incorrectly split by the clustering step. |

Multiple reviewers can work on the same project completely independently. Their decisions are stored in small per-reviewer overlay files; the large shared data is never duplicated.

---

## Environment Setup

Two virtual environments are required because TensorFlow 2.10 (needed for face attribute extraction) only supports Python 3.10, while the GUI itself runs on any modern Python.

### Why two environments?

| Env | Python | Used for |
|-----|--------|---------|
| `venv_gui` | 3.10 – 3.13 | Running the GUI, Tab 2 review, Tab 3 clustering, Tab 4 review |
| `venv_processing` | **3.10 exactly** | Tab 1 only — InsightFace detection + DeepFace attribute extraction |

Tab 3 (clustering) uses `faiss`, `igraph`, and `leidenalg` — no TensorFlow — so it runs inside `venv_gui` directly.

---

### 1. Create `venv_gui`

```bash
# From the project root — any Python 3.10+ works
python -m venv venv_gui

# Activate (Windows)
venv_gui\Scripts\activate

# Install
pip install -r requirements_gui.txt
```

`requirements_gui.txt` installs: `customtkinter`, `pandas`, `numpy`, `opencv-python`, `Pillow`, `faiss-cpu`, `python-igraph`, `leidenalg`, `networkx`, `scipy`.

---

### 2. Create `venv_processing`

**Python 3.10 is required.** TensorFlow 2.10 does not support 3.11 or later.

```bash
# Specify Python 3.10 explicitly (adjust path to your installation)
"C:\Python310\python.exe" -m venv venv_processing

# Activate (Windows)
venv_processing\Scripts\activate

# Install base dependencies
pip install -r requirements_processing.txt
```

`requirements_processing.txt` installs: `numpy<2`, `opencv-python`, `onnx`, `onnxruntime-gpu` (or `onnxruntime` for CPU), `deepface`, `tensorflow==2.10.*`.

#### Install InsightFace

InsightFace must be installed from a pre-built wheel because it does not have an official Windows pip release. Place the wheel in a `whls/` folder next to the project, then:

```bash
# Inside the activated venv_processing
pip install whls/insightface-0.7.3-cp310-cp310-win_amd64.whl
```

#### GPU support (optional)

For GPU inference, ensure CUDA 11.8 and cuDNN 8.x are installed, then use `onnxruntime-gpu` (already in `requirements_processing.txt`). For CPU-only, swap it for plain `onnxruntime`.

---

### 3. Configure the processing environment in the GUI

The GUI automatically searches for `venv_processing\Scripts\python.exe` and `venv_tf210\Scripts\python.exe` next to the project root. If your processing venv is in a different location:

1. Launch the GUI (see below).
2. In the startup dialog, expand **"Configure processing environment"**.
3. Browse to or paste the full path to the Python interpreter, e.g. `C:\path\to\venv_processing\Scripts\python.exe`.
4. Click **Continue** — the path is saved and remembered for future sessions.

---

## Running the GUI

Activate `venv_gui` first, then:

```bash
python run_gui.py
```

The startup dialog will ask for:
1. **Project directory** — the root folder containing participant sub-directories.
2. **Reviewer ID** — select an existing reviewer or create a new one. Reviewer profiles are stored in `{project}/_annotations/reviewers.json`.

---

## Project Directory Structure

```
ProjectRoot/
  _annotations/                          # all reviewer data (skipped when scanning for participants)
    reviewers.json                       # reviewer registry
    alice/
      Participant1/
        Session1/
          tab2_is_face.csv               # Tab 2: instance_index, is_face, reviewed_at
        tab3_face_ids.csv                # Tab 3: session_name, instance_index, face_id
        tab4_merges.csv                  # Tab 4: face_id, merged_face_id, reviewed_at
        final_faces_alice.csv            # on-demand export (Tab 4 → Export)
    bob/
      ...
  Participant1/
    Session1/
      scenevideo.mp4                     # input video (or .avi, .mov, ...)
      eye_tracking.tsv                   # optional gaze data
      face_detections.csv                # written by Tab 1, never modified afterwards
    Session2/
      ...
  Participant2/
    ...
```

---

## Tab-by-Tab Workflow

### Tab 1 — Face Detection

- Select sessions using the participant/session tree (supports multi-select).
- Configure sampling rate, GPU usage, and minimum detection confidence.
- Click **Start Processing**. Progress is streamed in real time.
- Requires `venv_processing` — see setup above.
- Output: `face_detections.csv` per session with columns:
  `frame_number, time_seconds, x, y, w, h, confidence, sharpness, distance, pitch, yaw, roll, attended, age, gender, race, emotion, embedding`

### Tab 2 — Face Instance Review

- Select a participant from the left panel, then a session from the right panel.
  Sessions that already have annotations show a ✓ marker.
- Click **Load Session** to display the face gallery.
- Click individual faces to toggle valid / non-face.
- Use bulk controls (select all, select by confidence, etc.) for efficiency.
- Click **Save Annotations** to write `tab2_is_face.csv` for the current reviewer.
- Annotations survive between visits — reload a session to resume where you left off.

### Tab 3 — Face ID Clustering

- Select participants to process.
- Configure clustering settings (algorithm, similarity threshold, k-neighbors, refinement).
- Click **Assign Face IDs**. The script reads all sessions' `face_detections.csv`
  and automatically applies the current reviewer's `tab2_is_face.csv` filters.
- Runs in `venv_gui` — no separate processing environment needed.
- Output: `tab3_face_ids.csv` — a thin overlay file (`session_name, instance_index, face_id`). Stats written to `tab3_stats.txt`.

### Tab 4 — Face ID Review

- Select a participant (only participants with completed Tab 3 results are listed).
- Click **Load Participant**. The GUI joins the base `face_detections.csv` files
  with the reviewer's `tab3_face_ids.csv` in memory — no combined CSV is ever written to disk.
- Browse face IDs in the list; double-click a face ID to open a gallery of all its instances.
- Select two or more face IDs and click **Merge** to combine them.
- Click **Save Annotations** to write `tab4_merges.csv`.
- Click **Export Final CSV** to produce `final_faces_{reviewer_id}.csv` — a full joined CSV written once on demand.

---

## Multi-Reviewer Workflow

1. Each reviewer runs the GUI, selects the shared project directory, and picks their reviewer ID from the startup dialog (or creates a new one).
2. Each reviewer works through Tabs 2 → 3 → 4 independently.
3. All decisions are stored exclusively in `_annotations/{reviewer_id}/`.
4. The large base files (`face_detections.csv`) are written once by Tab 1 and are **never modified** by any reviewer.
5. Final outputs per reviewer are small overlay files plus an on-demand export CSV.

---

## Key Files

| File | Purpose |
|------|---------|
| `run_gui.py` | Entry point |
| `gui_multitab.py` | Main GUI application (all four tabs + startup dialog) |
| `settings_manager.py` | GUI settings persistence + `ReviewerRegistry` class |
| `directory_tree_widget.py` | Reusable participant/session tree widget (Tab 1) |
| `stage1_detect_faces.py` | CLI script for face detection (called via subprocess, uses `venv_processing`) |
| `stage2_extract_attributes.py` | CLI script for attribute extraction (called via subprocess, uses `venv_processing`) |
| `stage3_graph_clustering.py` | CLI script for global face ID clustering (called via subprocess, uses `venv_gui`) |
| `video_processor.py` | Video-level processing utilities used by Stage 1 & 2 |
| `face_detection.py` | InsightFace detection helpers |
| `face_attributes.py` | DeepFace attribute extraction helpers |
| `utils.py` | Shared utility functions |
| `profiler.py` | Optional performance profiler |
| `requirements_gui.txt` | Dependencies for `venv_gui` |
| `requirements_processing.txt` | Dependencies for `venv_processing` |
