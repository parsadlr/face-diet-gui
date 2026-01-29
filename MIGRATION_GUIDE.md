# Migration Guide: Old GUI → New Multi-Tab GUI

This guide helps users familiar with the old single-purpose GUI (`gui.py`) transition to the new comprehensive multi-tab GUI (`gui_multitab.py`).

## Quick Start

**Old Way:**
```bash
# Run stages manually
python stage1_detect_faces.py session_dir
python stage2_extract_attributes.py session_dir
python stage3_graph_clustering.py participant_dir
python gui.py  # Then review and merge
```

**New Way:**
```bash
python run_gui.py  # All-in-one GUI
```

## Key Differences

### File Structure

| Old Files | New Files | Notes |
|-----------|-----------|-------|
| `stage1_detections.csv` | `face_detections.csv` | Stage 1 output |
| `stage2_attributes.csv` | `face_detections.csv` | Same file, updated |
| `faces_combined.csv` | `faces_combined.csv` | Unchanged |

**Why the change?**
- Reduced redundancy (one file instead of two per session)
- Clearer naming (`face_detections` instead of `stage1_detections`)
- Progressive updates (Stage 2 adds columns to existing file)

### Workflow Changes

#### Old Workflow
1. Run stage scripts manually (command line)
2. Check terminal output for errors
3. Navigate to participant folder
4. Open `gui.py` for review
5. Select folder, load data, review, merge

#### New Workflow
1. Open `run_gui.py` once
2. **Tab 1**: Process videos (replaces manual stage1/stage2)
3. **Tab 2**: Assign face IDs (replaces manual stage3)
4. **Tab 3**: Review and merge (enhanced version of old GUI)

All in one interface with better progress tracking!

## Feature Comparison

### Old GUI Features

| Feature | Old GUI | New GUI (Tab 3) | Enhancement |
|---------|---------|-----------------|-------------|
| Face ID list | ✓ | ✓ | Same |
| Thumbnails | ✓ | ✓ | Same |
| Merge/Unmerge | ✓ | ✓ | Same |
| Min instances filter | ✓ | ✓ | Same |
| Min confidence filter | ✓ | ✓ | Same |
| Session filtering | ✗ | ✓ | **NEW** |
| Gallery view | ✗ | ✓ | **NEW** |
| Double-click to view | ✗ | ✓ | **NEW** |
| Settings persistence | ✗ | ✓ | **NEW** |

### New Features Not in Old GUI

#### Session Filtering (Tab 3)
- **What**: Filter face IDs by selected sessions
- **Why**: Focus on specific sessions, exclude problematic ones
- **How**: Check/uncheck sessions, click "Apply Filter"

#### Gallery View (Tab 3)
- **What**: See all instances of a face ID at once
- **Why**: Verify all instances are the same person
- **How**: Double-click any face ID row

#### Video Processing Interface (Tab 1)
- **What**: GUI for running Stage 1 & 2
- **Why**: No more command line, better progress tracking
- **How**: Select sessions, adjust settings, click process

#### Face ID Assignment Interface (Tab 2)
- **What**: GUI for running Stage 3
- **Why**: Easier to adjust clustering parameters and see results
- **How**: Select participants, adjust settings, click assign

#### Settings Persistence
- **What**: All settings saved automatically
- **Why**: Don't re-enter settings every time
- **Where**: `~/.face_diet_config.json`

## Migrating Existing Projects

### If You Have Old CSV Files

**Scenario 1: Only have `stage1_detections.csv` and `stage2_attributes.csv`**

Option A (Quick):
```bash
# Just rename stage2_attributes.csv
cd session_dir
cp stage2_attributes.csv face_detections.csv
```

Option B (Clean):
```bash
# Re-run processing in new GUI (Tab 1)
# Will overwrite with new format
```

**Scenario 2: Have `faces_combined.csv` from old workflow**

No changes needed! Tab 3 works with existing `faces_combined.csv` files.

**Scenario 3: Have merged results with `merged_face_id` column**

Perfect! The new GUI respects existing merges.

### Data Compatibility

| Data Type | Compatible? | Notes |
|-----------|-------------|-------|
| Old stage1_detections.csv | Partial | Missing attribute columns |
| Old stage2_attributes.csv | Yes | Rename to face_detections.csv |
| Old faces_combined.csv | Yes | Works directly |
| Old merged_face_id column | Yes | Preserved on save |

## Script Usage vs GUI

### When to Use Scripts

You might still want to use command-line scripts for:
- **Automation**: Batch processing many participants
- **Server/HPC**: No GUI available
- **Reproducibility**: Script parameters easier to track
- **Integration**: Part of larger pipeline

The scripts still work! The GUI is just a convenient interface.

### When to Use GUI

Use the new GUI for:
- **Exploration**: Trying different parameters
- **Manual Review**: Inspecting and merging face IDs
- **Learning**: Understanding the pipeline
- **Convenience**: All-in-one interface
- **Progress Tracking**: Visual feedback

## Common Tasks

### Task: Process New Sessions

**Old Way:**
```bash
for session in Participant1/*; do
    python stage1_detect_faces.py "$session"
    python stage2_extract_attributes.py "$session"
done
python stage3_graph_clustering.py Participant1
```

**New Way:**
1. Open GUI
2. Tab 1: Select sessions, click "Start Processing"
3. Tab 2: Select participant, click "Assign Face IDs"

### Task: Adjust Clustering Parameters

**Old Way:**
```bash
# Edit script or use command line arguments
python stage3_graph_clustering.py Participant1 --threshold 0.65 --k-neighbors 75
# Check results, repeat if needed
```

**New Way:**
1. Open GUI, go to Tab 2
2. Adjust sliders/inputs in real-time
3. Click "Assign Face IDs"
4. View results immediately in Tab 3

### Task: Review and Merge Face IDs

**Old Way:**
```bash
python gui.py
# Browse to participant folder
# Load data
# Review and merge
# Save
```

**New Way:**
1. Open multi-tab GUI
2. Go to Tab 3 (or start from Tab 1 if processing fresh data)
3. Same review and merge workflow
4. Plus new features: session filtering, gallery view

### Task: Re-merge After Fixing Clustering

**Old Way:**
```bash
# Re-run stage3 with new parameters
python stage3_graph_clustering.py Participant1 --threshold 0.7
# Re-open gui.py
python gui.py
# Browse to participant again
# Start merging from scratch
```

**New Way:**
1. Tab 2: Adjust parameters, re-run clustering
2. Tab 3: Automatically updates, continue merging
3. Settings persist across sessions

## Troubleshooting Migration

### "faces_combined.csv not found" in Tab 3

**Cause**: Haven't run Tab 2 (Face ID Assignment) yet

**Solution**: 
1. Ensure sessions are processed (Tab 1)
2. Run Face ID Assignment (Tab 2) first
3. Then open Tab 3

### Old CSVs don't work in new GUI

**Cause**: Missing columns or wrong filename

**Solution**:
```bash
# Check what you have
ls session_dir/*.csv

# If you have stage2_attributes.csv, rename it
cp stage2_attributes.csv face_detections.csv

# Or re-run Tab 1 to regenerate
```

### Lost my merges after running Tab 2 again

**Cause**: `faces_combined.csv` was overwritten

**Solution**:
- Tab 2 creates fresh `faces_combined.csv` (no merged_face_id column)
- If you had merges, make sure you saved them first (Tab 3 → Save)
- The backup file `faces_combined.backup.csv` has your old merges

**Prevention**:
- Always save merges in Tab 3 before re-running Tab 2
- Or work on a copy of the participant directory

### Settings keep resetting

**Cause**: Config file not being saved

**Solution**:
- Check if `~/.face_diet_config.json` exists
- Check file permissions
- Try manually creating the file:
```bash
echo '{"last_project_dir": ""}' > ~/.face_diet_config.json
```

## Best Practices

### For New Users
1. Start with Tab 1, work through sequentially
2. Use default settings initially
3. Experiment with Tab 3 session filtering
4. Try gallery view to understand face IDs better

### For Experienced Users
1. Can jump directly to Tab 3 if data already processed
2. Use Tab 2 to quickly iterate on clustering parameters
3. Settings persistence saves time
4. Session filtering in Tab 3 is powerful for debugging

### For Large Projects
1. Process in batches (Tab 1: select subset of sessions)
2. Use Tab 2 one participant at a time
3. Save Tab 3 merges frequently
4. Keep backups of `faces_combined.csv`

## Keyboard Shortcuts

| Action | Old GUI | New GUI |
|--------|---------|---------|
| Open gallery | N/A | Double-click face ID row |
| Merge | Click button | Click button |
| Unmerge | Click button | Click button |
| Select all | Manual | Checkbox |
| Filter sessions | N/A | Session filter panel |

## Performance Comparison

| Task | Old GUI | New GUI |
|------|---------|---------|
| Load face IDs | Same | Same |
| Extract thumbnails | Same | Same |
| Gallery view | N/A | New feature |
| Session filtering | N/A | Instant (already loaded) |
| Settings load | N/A | Instant (auto-loaded) |

## Getting Help

If you're stuck:

1. Check `GUI_README.md` for detailed docs
2. Check `TESTING_CHECKLIST.md` for common workflows
3. Review this migration guide
4. Check terminal output for errors
5. Verify file structure matches expectations

## FAQ

**Q: Can I still use the old scripts?**
A: Yes! They still work. The GUI just provides a convenient interface.

**Q: Do I need to reprocess everything?**
A: No, but renaming `stage2_attributes.csv` to `face_detections.csv` is recommended.

**Q: Will my old merges be preserved?**
A: Yes, if you have `merged_face_id` column in `faces_combined.csv`, Tab 3 will respect it.

**Q: Is the new GUI slower?**
A: No, processing speed is the same. GUI adds convenience but doesn't change algorithms.

**Q: Can I use both old and new GUI?**
A: Yes, but be careful with file naming. Old GUI expects old filenames.

**Q: What happened to the old gui.py?**
A: It's still there! You can use it if you prefer. The new one is `gui_multitab.py`.

## Summary

The new multi-tab GUI:
- ✓ Includes all old GUI features
- ✓ Adds new features (session filtering, gallery view)
- ✓ Provides interface for entire pipeline
- ✓ Maintains compatibility with existing data
- ✓ Improves user experience

**Bottom Line**: The new GUI is a superset of the old one. You can do everything you could before, plus more!
