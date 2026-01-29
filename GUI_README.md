# Face-Diet Multi-Tab GUI

A comprehensive GUI for the entire face processing pipeline, from video processing to manual review and merging.

## Overview

The new multi-tab GUI provides a complete interface for:
1. **Video Processing** - Face detection and attribute extraction (Stages 1 & 2)
2. **Face ID Assignment** - Global face clustering across sessions (Stage 3)
3. **Manual Review & Merging** - Review and manually merge face IDs

## Getting Started

### Requirements

- Python 3.10+
- All dependencies from `requirements.txt`
- Additional GUI dependency: `customtkinter`

```bash
pip install customtkinter
```

### Running the GUI

```bash
python run_gui.py
```

Or directly:

```bash
python gui_multitab.py
```

## Tab 1: Video Processing

Process video files to detect faces and extract attributes.

### Features

- **Directory Tree Browser**: Navigate your project structure
  - Select specific participants and sessions
  - View file status (video, eye tracking)
- **Stage 1 Settings**: 
  - Sampling rate (frames to skip)
  - GPU acceleration
- **Stage 2 Settings**:
  - Batch size for attribute extraction
- **Progress Tracking**:
  - Real-time status updates
  - Detailed processing log
  - Progress bar

### Workflow

1. Click **Browse** to select your project root directory
2. The directory tree will show all participants and sessions
3. Use **Expand All** / **Collapse All** to navigate
4. Use **Select All** / **Deselect All** to choose which sessions to process
5. Adjust settings as needed
6. Click **Start Processing**

### Output

- Creates `face_detections.csv` in each session directory
- Contains face detections (bounding boxes, embeddings, pose) and attributes (age, gender, emotion, etc.)

### Expected Directory Structure

```
Project/
  Participant1/
    Session1/
      scenevideo.mp4
      eye_tracking.tsv (optional)
    Session2/
      ...
  Participant2/
    ...
```

## Tab 2: Face ID Assignment

Assign global face IDs using graph-based clustering.

### Features

- **Participant Selection**: Choose which participants to process
- **Clustering Settings**:
  - Algorithm (Leiden or Louvain)
  - Similarity threshold
  - k-neighbors
  - Min confidence filter
  - Small cluster refinement options
- **Progress Tracking**: Real-time updates and detailed logs

### Workflow

1. Click **Browse** to select project root (or it will use the path from Tab 1)
2. The participant list will show all participants with processed sessions
3. Select which participants to process
4. Adjust clustering settings as needed
5. Click **Assign Face IDs**

### Output

- Creates `faces_combined.csv` in each participant directory
- Contains all face instances with assigned global `face_id`
- Also creates `faces_combined.stats.txt` with clustering statistics

### Settings Explained

- **Similarity Threshold**: Minimum cosine similarity for faces to be connected (0.0-1.0)
- **k-Neighbors**: Number of nearest neighbors to consider for each face
- **Min Confidence**: Filter out low-confidence detections
- **Enable Refinement**: Post-process small clusters by reassigning to larger clusters
- **Min Cluster Size**: Clusters with ≤ this many faces are candidates for refinement

## Tab 3: Manual Review & Merging

Review face IDs and manually merge incorrect clusters.

### Features

- **Session Filtering**: Choose which sessions to include in review
- **Face ID List**: 
  - View all detected face IDs
  - See representative thumbnail
  - View instance counts
  - Filter by min instances and confidence
- **Gallery View**: Double-click any face ID to see all instances
- **Merging**: Select multiple face IDs and merge them
- **Unmerging**: Undo merges if needed
- **Save Results**: Add `merged_face_id` column to CSV

### Workflow

1. Click **Browse** to select a participant directory
2. Click **Review** to load the data
3. (Optional) Use session filter to focus on specific sessions
4. Review face IDs:
   - Double-click to open gallery view
   - Select multiple IDs that represent the same person
   - Click **Merge Selected**
5. Continue until satisfied
6. Click **Save Merged Results**

### Gallery View

- Shows all instance crops for a selected face ID
- Scrollable grid layout
- Helpful for verifying if all instances are the same person

### Output

- Updates `faces_combined.csv` with new `merged_face_id` column
- Creates backup: `faces_combined.backup.csv`
- `merged_face_id` contains the final face ID after manual merging
- For faces not merged, `merged_face_id` equals `face_id`

## Settings Persistence

All settings are automatically saved to `~/.face_diet_config.json` and restored when you reopen the GUI:

- Last used project directory
- Stage 1 & 2 settings
- Stage 3 clustering settings
- Tab 3 filter settings

## File Naming Changes

The new workflow uses simplified file naming:

- **Old**: `stage1_detections.csv` → `stage2_attributes.csv` → `faces_combined.csv`
- **New**: `face_detections.csv` (updated progressively) → `faces_combined.csv`

Benefits:
- Less redundancy
- Clearer purpose
- Easier to manage

## Keyboard Shortcuts & Tips

- **Tab 3**: Double-click any face ID row to open gallery
- **Session Filter**: Use to focus on problematic sessions
- **Merge Strategy**: Start with high-instance faces (usually participants), then handle smaller clusters
- **Gallery View**: Check boundary cases - faces at different angles or lighting

## Troubleshooting

### "No session directories found"
- Ensure your project structure matches the expected format
- Check that directories aren't hidden (starting with `.`)

### "No face_detections.csv files found"
- Run Tab 1 (Video Processing) first
- Ensure processing completed successfully

### "No faces_combined.csv found"
- Run Tab 2 (Face ID Assignment) first
- Check that clustering completed successfully

### GUI freezes during processing
- This should not happen (threading is used)
- If it does, check the terminal for error messages

### Gallery view shows wrong faces
- Check that `face_detections.csv` has correct session names
- Verify video files are in the expected locations

## Technical Details

### Architecture

- **Settings Manager**: JSON-based configuration persistence
- **Directory Tree Widget**: Custom CTk widget for hierarchical navigation
- **Progress Reporter**: Thread-safe progress updates
- **Tab Classes**: Separate classes for each tab (modular design)

### Threading

All heavy processing runs in background threads to keep the GUI responsive:
- Tab 1: Session-by-session processing
- Tab 2: Participant-by-participant clustering
- Tab 3: Image loading for gallery view

### Face Crop Extraction

The GUI extracts face crops directly from video files:
- Uses OpenCV for video reading
- Caches per-session bbox statistics
- Handles missing videos gracefully (shows placeholder)

## Known Limitations

1. **Gallery View**: Loading hundreds of images may take a few seconds
2. **Large Datasets**: Very large participant datasets (100K+ faces) may be slow
3. **Video Codecs**: Some video formats may not be supported by OpenCV

## Comparison with Original GUI

### Old GUI (`gui.py`)
- Single purpose: Face ID review and merging only
- Required manual CSV preparation
- No session filtering
- No gallery view

### New GUI (`gui_multitab.py`)
- Complete pipeline: Processing → Clustering → Review
- Integrated workflow across tabs
- Session filtering in Tab 3
- Gallery view for detailed inspection
- Settings persistence
- Better progress tracking

## Future Enhancements

Potential improvements:
- Parallel processing of multiple sessions
- Export merged IDs to various formats
- Face ID statistics and visualizations
- Batch unmerge operations
- Search/filter face IDs by criteria

## Support

For issues or questions:
1. Check this README
2. Review terminal output for error messages
3. Check `~/.face_diet_config.json` for settings issues
4. Verify file structure and naming conventions
