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

## Requirements

```
Python 3.9+
customtkinter
pandas
numpy
opencv-python
Pillow
faiss-cpu          # for Stage 3 clustering
python-igraph      # for Leiden algorithm
leidenalg          # for Leiden algorithm
networkx           # fallback for Louvain
scipy
deepface           # for Stage 2 attribute extraction
insightface        # for Stage 1 face detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Stage 1 and Stage 3 are run via a separate virtual environment (`venv_tf210`) to isolate TensorFlow dependencies. Create it and install dependencies there as well.

---

## Running the GUI

```bash
python run_gui.py
```

A startup dialog will appear asking for:
1. **Project directory** — the root folder containing participant sub-directories.
2. **Reviewer ID** — select an existing reviewer or create a new one. Reviewer profiles are saved in `{project}/_annotations/reviewers.json`.

---

## Project Directory Structure

```
ProjectRoot/
  _annotations/                          # all reviewer data (never confused with participants)
    reviewers.json                       # reviewer registry
    alice/
      progress.json                      # optional progress metadata
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

The `_annotations/` directory is automatically skipped when the GUI scans for participants and sessions.

---

## Tab-by-Tab Workflow

### Tab 1 — Face Detection

- Select sessions using the participant/session tree (supports multi-select).
- Configure sampling rate, GPU usage, and minimum detection confidence.
- Click **Start Processing**. Progress is streamed in real time.
- Output: `face_detections.csv` per session with columns:
  `frame_number, time_seconds, x, y, w, h, confidence, sharpness, distance,`
  `pitch, yaw, roll, attended, age, gender, race, emotion, embedding`

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
- Output: `tab3_face_ids.csv` — a thin overlay file containing only
  `session_name, instance_index, face_id` (not a copy of the full data).
- A `tab3_stats.txt` summary is also written alongside.

### Tab 4 — Face ID Review

- Select a participant (only participants with completed Tab 3 results are listed).
- Click **Load Participant**. The GUI joins the base `face_detections.csv` files
  with the reviewer's `tab3_face_ids.csv` in memory.
- Browse face IDs in the list; double-click a face ID to open a gallery of all
  its instances.
- Select two or more face IDs and click **Merge** to combine them.
- Click **Save Annotations** to write `tab4_merges.csv`.
- Click **Export Final CSV** to produce `final_faces_{reviewer_id}.csv` — a full
  CSV joining base data + face IDs + merges, written once on demand.

---

## Multi-Reviewer Workflow

1. Each reviewer runs the GUI, selects the shared project directory, and picks
   their reviewer ID from the startup dialog (or creates a new one).
2. Each reviewer works through Tabs 2 → 3 → 4 independently.
3. All decisions are stored exclusively in `_annotations/{reviewer_id}/`.
4. The large base files (`face_detections.csv`) are written once by Tab 1 and
   are **never modified** by any reviewer.
5. Final outputs per reviewer are thin overlay files plus an on-demand export CSV.

---

## Key Files

| File | Purpose |
|------|---------|
| `run_gui.py` | Entry point |
| `gui_multitab.py` | Main GUI application (all four tabs + startup dialog) |
| `settings_manager.py` | GUI settings persistence + `ReviewerRegistry` class |
| `directory_tree_widget.py` | Reusable participant/session tree widget (Tab 1) |
| `stage1_detect_faces.py` | CLI script for face detection (called via subprocess) |
| `stage2_extract_attributes.py` | CLI script for attribute extraction (called via subprocess) |
| `stage3_graph_clustering.py` | CLI script for global face ID clustering (called via subprocess) |
| `video_processor.py` | Video-level processing utilities used by Stage 1 & 2 |
| `face_detection.py` | InsightFace detection helpers |
| `face_attributes.py` | DeepFace attribute extraction helpers |
| `utils.py` | Shared utility functions |
| `profiler.py` | Optional performance profiler |
