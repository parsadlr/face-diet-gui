# Face-Diet

A GUI application for processing egocentric (first-person) video to detect, review, and identify faces seen by the camera wearer. Designed for multi-reviewer research workflows where several annotators work on the same dataset independently, then reconcile disagreements.

---

## Pipeline Overview

The workflow combines three automated processing stages with two manual review steps.

```
Video files
    │
    ▼
[Tab 1]  Face Detection          (InsightFace)
         → bounding boxes, embeddings, pose, attended flag
    │
    ▼
[Tab 1]  Attribute Extraction    (DeepFace)
         → age, gender, race, emotion appended to detections
    │
    ▼
[Tab 2]  Manual Review — Face Instance Review
         → per-reviewer valid/non-face labels
    │
    ▼
[Tab 3]  Mismatch Resolution     (multi-reviewer)
         → reconcile disagreements across reviewers
    │
    ▼
[Tab 4]  Face ID Clustering      (FAISS k-NN + Louvain/Leiden)
         → global face IDs assigned across all sessions
    │
    ▼
[Tab 5]  Manual Review — Face ID Review
         → merge/correct face IDs
```

---

## ⚙️ Setup

### Requirements

- **Python 3.10** — required by TensorFlow 2.10 (used in attribute extraction) and InsightFace.

### Set Python version

The project includes a `.python-version` file for pyenv/asdf. Use Python 3.10 when creating the venv:

**With pyenv:**
```bash
pyenv install 3.10.14   # if not installed
pyenv local 3.10        # uses .python-version
```

**Without a version manager:**
```bash
# macOS (Homebrew): brew install python@3.10
# Ubuntu: sudo apt install python3.10 python3.10-venv
# Then use the full path or python3.10:
python3.10 -m venv venv
```

### Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### InsightFace on Windows

InsightFace has no official pip wheel for Windows. Install from a local build:

```bash
pip install whls/insightface-0.7.3-cp310-cp310-win_amd64.whl
```

Place the `.whl` file in a `whls/` folder at the project root. Pre-built wheels for Python 3.10 on Windows can be found in the InsightFace community releases.

### GPU vs CPU

`requirements.txt` installs `onnxruntime-gpu` (for ONNX-based face detection). To run on CPU only, open `requirements.txt` and swap:

```
# Comment out:
onnxruntime-gpu==1.18.1
# Uncomment:
# onnxruntime>=1.16.0
```

---

## ▶️ Running

Activate the virtual environment first, then run from the project root:

```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

python main.py
# or
python -m face_diet_gui
```

### Startup dialog

On launch a setup dialog appears asking for:

- **Project directory** — the root folder containing participant subfolders.
- **Reviewer ID** — select an existing reviewer or type a new name to create one. The reviewer list is stored in `{project_dir}/_annotations/reviewers.json` and is shared across all users of the same project directory.

The last-used values are remembered across sessions (`~/.face_diet_config.json`).

---

## Using the GUI

### Tab 1 — Face Detection & Attribute Extraction

Runs face detection and optionally attribute extraction via subprocess, one session at a time or in parallel.

| Setting | Description |
|---|---|
| Sampling rate | Process every N frames (e.g. 30 = 1 fps for 30 fps video) |
| Min confidence | Filter out low-confidence detections (0.0–1.0) |
| GPU | Use ONNX GPU runtime for faster detection |
| Start / end time | Restrict processing to a time window |

**Output per session:** `face_detections.csv` — bounding boxes, 512-dim face embeddings, yaw/pitch/roll pose angles, and an `attended` flag (derived from eye-tracking data if `eye_tracking.tsv` is present). Attribute extraction (age, gender, race, emotion) updates this same file in-place.

> Re-running face detection for a session automatically invalidates and removes that session's existing reviewer annotations (face/non-face labels and manual merges) to keep data consistent.

---

### Tab 2 — Face Instance Review

Manual review of every detected face crop in a session. The reviewer marks each detection as **valid face** or **non-face** (e.g. poster, photo, partial detection).

Labels are saved as a per-reviewer overlay and never modify the base detection CSV, so multiple reviewers can label the same session independently.

**Output:** `_annotations/{reviewer_id}/{participant}/{session}/is_face.csv`

---

### Tab 3 — Mismatch Resolution

When two or more reviewers have labeled the same session, this tab highlights detections where they disagree. The current reviewer can inspect each mismatch and cast a deciding vote, producing a shared consensus label used downstream by clustering.

**Output:** `_annotations/consensus/{participant}/{session}/consensus_is_face.csv` — stored in a shared `consensus/` directory (not per-reviewer).

---

### Tab 4 — Face ID Clustering

Runs graph-based community detection for a selected participant. Loads all sessions' embeddings, builds a k-NN similarity graph with FAISS, enforces a same-frame constraint (two faces in the same frame cannot be the same person), then runs community detection to assign a global face ID to every detection.

| Setting | Description |
|---|---|
| Similarity threshold | Cosine similarity edge threshold for k-NN graph |
| k neighbors | Number of nearest neighbors per node |
| Algorithm | Leiden (default, higher quality) or Louvain |
| Enable refinement | Re-assign small clusters via k-NN voting |

**Output:** `{participant}/face_ids.csv` and `{participant}/clustering_stats.txt` — written directly to the participant folder and shared across reviewers.

---

### Tab 5 — Face ID Review

Manual review and correction of the clustering output. The reviewer can browse face IDs, view sample crops, and merge two IDs that the algorithm split incorrectly.

**Output:** `_annotations/{reviewer_id}/{participant}/merges.csv` — merge decisions and media flags.

---

## 🗂️ Data Directory Structure

```
project_root/
├── _annotations/
│   ├── reviewers.json                        ← shared reviewer registry
│   ├── alice/                                ← per-reviewer overlays
│   │   └── participant_01/
│   │       ├── session_a/
│   │       │   ├── is_face.csv               ← Tab 2: face/non-face labels
│   │       │   └── review_status.json        ← Tab 2: reviewed flag
│   │       └── merges.csv                    ← Tab 5: ID merge decisions
│   ├── bob/
│   │   └── ...
│   └── consensus/                            ← shared across reviewers (Tab 3)
│       └── participant_01/
│           └── session_a/
│               ├── consensus_is_face.csv     ← Tab 3: resolved consensus labels
│               └── mismatches_resolved.json  ← Tab 3: resolution flag
├── participant_01/
│   ├── face_ids.csv                          ← Tab 4 clustering output (shared)
│   ├── clustering_stats.txt                  ← Tab 4 clustering stats (shared)
│   ├── session_a/
│   │   ├── scenevideo.mp4                    ← required: source video
│   │   ├── eye_tracking.tsv                  ← optional: gaze data
│   │   └── face_detections.csv               ← Tab 1 output (shared base data)
│   └── session_b/
│       └── ...
└── participant_02/
    └── ...
```

The GUI skips `_annotations/` when scanning for participants.

---

## 📋 Output Files Reference

| File | Location | Written by | Contents |
|---|---|---|---|
| `face_detections.csv` | `{participant}/{session}/` | Tab 1 | Bounding boxes, embeddings, pose, attended flag; demographic attributes added in-place by attribute extraction |
| `is_face.csv` | `_annotations/{reviewer}/{participant}/{session}/` | Tab 2 | Per-detection face/non-face label |
| `review_status.json` | `_annotations/{reviewer}/{participant}/{session}/` | Tab 2 | `{"reviewed": true/false}` flag |
| `consensus_is_face.csv` | `_annotations/consensus/{participant}/{session}/` | Tab 3 | Consensus label after mismatch resolution (shared) |
| `mismatches_resolved.json` | `_annotations/consensus/{participant}/{session}/` | Tab 3 | Flag marking when all mismatches are resolved (shared) |
| `face_ids.csv` | `{participant}/` | Tab 4 | Global face ID per detection (shared across reviewers) |
| `clustering_stats.txt` | `{participant}/` | Tab 4 | Clustering statistics (shared across reviewers) |
| `merges.csv` | `_annotations/{reviewer}/{participant}/` | Tab 5 | Manual ID merge decisions and media flags |

`face_detections.csv` is the shared base data — written by face detection, updated in-place by attribute extraction, and never modified by the review workflow. `face_ids.csv` and `clustering_stats.txt` are also shared (written to the participant folder directly). All reviewer-specific decisions live under `_annotations/{reviewer_id}/`.

---

## 👥 Multi-Reviewer Workflow

1. All reviewers point the app at the **same project directory** on a shared drive (or each work on a local copy and merge later).
2. Face detection and attribute extraction are run once — their outputs are shared base data.
3. Each reviewer completes Tab 2 independently, writing to their own subdirectory under `_annotations/`.
4. Tab 3 computes pairwise mismatches and lets each reviewer resolve disagreements.
5. Tab 4 clustering respects each reviewer's consensus annotations when filtering detections.
6. Tab 5 merge decisions are per-reviewer.

---

## Repo Layout

```
face-diet/
├── main.py                          ← entry point: python main.py
├── requirements.txt                 ← single venv for GUI + all processing
├── README.md
├── face_diet_gui/                   ← main package (python -m face_diet_gui)
│   ├── core/
│   │   ├── settings_manager.py      ← SettingsManager + ReviewerRegistry
│   │   └── pipeline_helpers.py      ← subprocess stage runners, session helpers
│   ├── gui/
│   │   ├── app.py                   ← StartupDialog + FaceDietApp main window
│   │   ├── common.py                ← shared GUI helpers (ProgressReporter, etc.)
│   │   ├── tabs/                    ← one file per tab (tab1_–tab5_)
│   │   └── widgets/
│   │       └── directory_tree_widget.py
│   ├── processing/
│   │   ├── video_processor.py       ← frame sampling, detection collection
│   │   ├── face_detection.py        ← InsightFace detector initialisation
│   │   └── face_attributes.py       ← DeepFace attribute extraction
│   ├── stages/                      ← scripts invoked via subprocess by the GUI
│   │   ├── detect_faces.py          ← face detection (InsightFace)
│   │   ├── extract_attributes.py    ← attribute extraction (DeepFace)
│   │   └── cluster_face_ids.py      ← graph-based face ID clustering
│   ├── utils.py                     ← blur score, pose frontality, CSV helpers
│   └── profiler.py                  ← optional performance profiling
└── .cursor/
    └── plans/                       ← AI planning artifacts (not part of the app)
```

The scripts under `face_diet_gui/stages/` are designed to be run in a separate subprocess (possibly under a different Python interpreter) so that heavy ML dependencies are isolated from the GUI process. They can also be run directly from the command line for debugging.
