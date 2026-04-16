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

- **Data directory** — root of your raw BIDS-style layout (`sub-XX/ses-YY/…` with `scenevideo.*` per session). Used to read videos and optional `eye_tracking.tsv`.
- **Derivatives directory** — root where pipeline outputs and all annotations are written. The shared reviewer registry lives at `{derivatives}/annotations/reviewers.json`.
- **Reviewer ID** — select an existing reviewer or type a new name to create one.

The last-used directories and reviewer are remembered across sessions (`~/.face_diet_config.json`).

---

## Using the GUI

### Tab 1 — Face Detection & Attribute Extraction

Runs face detection and optionally attribute extraction via subprocess, one session at a time or in parallel.

| Setting | Description |
|---|---|
| Exclude edges | Skip the first *N* and/or last *M* seconds of every video before processing. Useful to exclude the experimenter's setup phase where their face typically appears. Both fields are optional — leave either blank to skip only one side. Applies to both normal mode and interval sampling (the random search window is confined to the trimmed range). |
| Downsampling | When enabled, run detection on every *N*th frame (`factor`); when off, every frame is considered. |
| Interval sampling | Optional alternative to scanning the whole video: randomly pick up to *N* non-overlapping windows of *L* seconds that pass a quick face-density pre-check (`min face fraction`). After each accepted interval, up to 50 additional candidate windows are tried before raising an error. |
| Min confidence | Filter out low-confidence detections (0.0–1.0) |
| GPU | Use ONNX GPU runtime for faster detection |
| Batch size | Stage 2 (DeepFace) batch size |

All numeric fields are empty by default; leave any field blank to use the built-in fallback (factor: 3, interval length: 30 s, num. intervals: 5, min face fraction: 0.1, min confidence: 0.0, batch size: 32).

**Output per session (under derivatives):** `{participant}_{session}_face-detections.csv` — bounding boxes, 512-dim embeddings, pose, `attended` (if `eye_tracking.tsv` is present). Attribute extraction updates this file in-place. Videos are read from the **data** directory; CSVs are written under **derivatives** with BIDS-style filenames.

> Re-running face detection for a session automatically invalidates and removes that session's existing reviewer annotations (face/non-face labels and manual merges) to keep data consistent.

---

### Tab 2 — Face Instance Review

Manual review of every detected face crop in a session. The reviewer marks each detection as **valid face** or **non-face** (e.g. poster, photo, partial detection).

Labels are saved as a per-reviewer overlay and never modify the base detection CSV, so multiple reviewers can label the same session independently.

**Output:** `{derivatives}/annotations/{reviewer_id}/{participant}/{session}/{participant}_{session}_is-face.csv`

---

### Tab 3 — Mismatch Resolution

When two or more reviewers have labeled the same session, this tab highlights detections where they disagree. The current reviewer can inspect each mismatch and cast a deciding vote, producing a shared consensus label used downstream by clustering.

**Output:** `{derivatives}/annotations/consensus/{participant}/{session}/{participant}_{session}_consensus-is-face.csv` (shared, not per-reviewer).

---

### Tab 4 — Face ID Clustering

Runs graph-based community detection for a selected participant. Loads all sessions' embeddings, builds a k-NN similarity graph with FAISS, enforces a same-frame constraint (two faces in the same frame cannot be the same person), then runs community detection to assign a global face ID to every detection.

| Setting | Description |
|---|---|
| Similarity threshold | Cosine similarity edge threshold for k-NN graph |
| k neighbors | Number of nearest neighbors per node |
| Algorithm | Leiden (default, higher quality) or Louvain |
| Enable refinement | Re-assign small clusters via k-NN voting |

**Output:** `{derivatives}/{participant}/{participant}_face-ids.csv` and `{derivatives}/{participant}/{participant}_clustering-stats.txt` — shared across reviewers.

---

### Tab 5 — Face ID Review

Manual review and correction of the clustering output. The reviewer can browse face IDs, view sample crops, and merge two IDs that the algorithm split incorrectly.

**Output:** `{derivatives}/annotations/{reviewer_id}/{participant}/merges.csv` — merge decisions and media flags.

---

## 🗂️ Data layout (two roots)

**Data directory (raw):** participant / session folders with media only — e.g. `sub-01/ses-01/scenevideo.mp4` and optional `eye_tracking.tsv`. Nothing under `annotations/`.

**Derivatives directory (outputs):** everything the app writes — BIDS-style CSV names, plus `annotations/` for reviewers and consensus.

```
data_dir/
└── sub-01/
    └── ses-01/
        ├── scenevideo.mp4
        └── eye_tracking.tsv              ← optional

derivatives_dir/
├── annotations/
│   ├── reviewers.json
│   ├── alice/                          ← per-reviewer overlays
│   │   └── sub-01/
│   │       ├── ses-01/
│   │       │   ├── sub-01_ses-01_is-face.csv
│   │       │   └── sub-01_ses-01_review-status.json
│   │       └── merges.csv              ← Tab 5
│   └── consensus/                      ← Tab 3 (shared)
│       └── sub-01/
│           └── ses-01/
│               ├── sub-01_ses-01_consensus-is-face.csv
│               └── sub-01_ses-01_mismatches-resolved.json
└── sub-01/
    ├── sub-01_face-ids.csv             ← Tab 4 (shared)
    ├── sub-01_clustering-stats.txt
    └── ses-01/
        └── sub-01_ses-01_face-detections.csv   ← Tab 1 (shared base)
```

The session tree in Tab 1 is built from the **data** directory; detection status and downstream tabs use paths under **derivatives**.

### Eye tracking export (Tobii Pro Lab)

Face detection reads gaze from **`eye_tracking.tsv`** in each session folder (next to `scenevideo.*`). To export from **Tobii Pro Lab**:

1. Add your recordings to a Tobii Pro Lab project.
2. Open **Data export** (top right).
3. Under **Data fields**, enable the **Eye tracking data** group.
4. Set **Format** to **Multiple standard files (.tsv)**, which will export one file per recording session.
5. Leave **Export units** disabled (unchecked) so gaze columns stay **`Gaze point X`** / **`Gaze point Y`** without `[MCS px]` suffixes.
6. Set **Timestamp precision** to **microseconds**.
7. Set **Gaze filter** to **Tobii I-VT (Attention)**.
8. Enable **Recording gaze data**.
9. Run the export. You get a separate `.tsv` per recording; copy or rename each into the correct **`data_dir/{participant}/{session}/`** folder as **`eye_tracking.tsv`** so the GUI and detection stage can find it (rename if Tobii’s default filename differs).

The pipeline requires these header names (exact match after trimming): **`Sensor`**, **`Gaze point X`**, **`Gaze point Y`**, and a recording timestamp column **`Recording timestamp`** or **`Recording timestamp [ms]`**. Rows with **`Sensor`** = **`Eye Tracker`** are used for gaze.

---

## 📋 Output Files Reference

Paths are relative to **derivatives**. `{p}` = participant ID, `{s}` = session ID.

| File | Location | Written by | Contents |
|---|---|---|---|
| `{p}_{s}_face-detections.csv` | `{p}/{s}/` | Tab 1 | Bounding boxes, embeddings, pose, attended; attributes updated in-place in stage 2 |
| `{p}_{s}_is-face.csv` | `annotations/{reviewer}/{p}/{s}/` | Tab 2 | Per-detection face/non-face label |
| `{p}_{s}_review-status.json` | `annotations/{reviewer}/{p}/{s}/` | Tab 2 | `{"reviewed": true/false}` |
| `{p}_{s}_consensus-is-face.csv` | `annotations/consensus/{p}/{s}/` | Tab 3 | Consensus after mismatch resolution (shared) |
| `{p}_{s}_mismatches-resolved.json` | `annotations/consensus/{p}/{s}/` | Tab 3 | Resolution flag (shared) |
| `{p}_face-ids.csv` | `{p}/` | Tab 4 | Global face ID per detection (shared) |
| `{p}_clustering-stats.txt` | `{p}/` | Tab 4 | Clustering statistics (shared) |
| `merges.csv` | `annotations/{reviewer}/{p}/` | Tab 5 | ID merges and media flags |

Face-detections CSV is the shared base for review tabs. Reviewer-specific files live under `annotations/{reviewer}/`. Stage scripts still accept legacy names (`face_detections.csv`, etc.) where noted in their CLI help.

---

## 👥 Multi-Reviewer Workflow

1. All reviewers use the **same derivatives directory** (and typically the same data directory) on a shared drive, or merge those trees later.
2. Face detection and attribute extraction write shared base CSVs under derivatives — run once per session unless you intentionally re-run.
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
