# Face-Diet Multi-Tab GUI - Testing Checklist

Use this checklist to verify the GUI implementation works correctly.

## Pre-Testing Setup

- [ ] Install customtkinter: `pip install customtkinter`
- [ ] Have a test project directory with:
  - At least one participant folder
  - At least one session with `scenevideo.*` file
  - Optional: `eye_tracking.tsv` file

## General Tests

### Application Launch
- [ ] Launch GUI: `python run_gui.py`
- [ ] GUI opens without errors
- [ ] Window title is correct: "Face-Diet: Comprehensive Face Processing Pipeline"
- [ ] All 3 tabs are visible: "Video Processing", "Face ID Assignment", "Manual Review"
- [ ] Dark theme is applied

### Settings Persistence
- [ ] Change some settings in each tab
- [ ] Close and reopen the GUI
- [ ] Verify settings are restored
- [ ] Check `~/.face_diet_config.json` exists and contains settings

## Tab 1: Video Processing

### UI Components
- [ ] Browse button works
- [ ] Directory entry is readonly
- [ ] Settings panel shows Stage 1 and Stage 2 options
- [ ] Progress section has status label, progress bar, and log textbox
- [ ] "Start Processing" button is present

### Directory Tree Widget
- [ ] Browse and select project directory
- [ ] Directory tree displays participants and sessions
- [ ] Sessions show file status (✓ video, ✓ eye tracking)
- [ ] Expand/Collapse All buttons work
- [ ] Select All/Deselect All buttons work
- [ ] Can manually check/uncheck individual sessions
- [ ] Parent checkbox controls child checkboxes

### Processing Workflow
- [ ] Select a single session
- [ ] Adjust sampling rate (try 30)
- [ ] Click "Start Processing"
- [ ] Button becomes disabled during processing
- [ ] Status label updates ("Processing X/Y...")
- [ ] Progress bar updates
- [ ] Log shows detailed messages
- [ ] Processing completes without errors
- [ ] `face_detections.csv` created in session directory
- [ ] CSV contains expected columns: face_id, session_name, frame_number, x, y, w, h, embedding, age, gender, etc.

### Multiple Sessions
- [ ] Select multiple sessions (2-3)
- [ ] Process them
- [ ] Verify each session gets its own `face_detections.csv`
- [ ] Log shows progress for each session

### Error Handling
- [ ] Try processing without selecting a directory (should show error)
- [ ] Try processing without selecting any sessions (should show error)
- [ ] Try processing a session without video file (should handle gracefully)

## Tab 2: Face ID Assignment

### UI Components
- [ ] Browse button works (or inherits from Tab 1)
- [ ] Participant list is empty before browsing
- [ ] Settings panel shows all clustering parameters
- [ ] Progress section is present
- [ ] "Assign Face IDs" button is present

### Participant List
- [ ] Browse to project directory (that has processed sessions from Tab 1)
- [ ] Participant list shows participants with session counts
- [ ] Only shows participants with `face_detections.csv` files
- [ ] Select All/Deselect All buttons work
- [ ] Can manually check/uncheck participants

### Processing Workflow
- [ ] Select a participant
- [ ] Use default settings (or adjust as needed)
- [ ] Click "Assign Face IDs"
- [ ] Button becomes disabled during processing
- [ ] Status label updates
- [ ] Progress bar updates
- [ ] Log shows detailed clustering progress
- [ ] Processing completes without errors
- [ ] `faces_combined.csv` created in participant directory
- [ ] `faces_combined.stats.txt` created
- [ ] CSV contains face_id column with "FACE_XXXXX" format

### Settings Verification
- [ ] Try different similarity thresholds (0.5, 0.6, 0.7)
- [ ] Try different k-neighbors values (25, 50, 100)
- [ ] Toggle refinement on/off
- [ ] Verify results differ as expected

### Error Handling
- [ ] Try without selecting directory (should show error)
- [ ] Try without selecting any participants (should show error)
- [ ] Try participant without processed sessions (should not appear in list)

## Tab 3: Manual Review & Merging

### UI Components
- [ ] Browse button works
- [ ] Session filter panel is hidden initially
- [ ] Filter controls (Min Instances, Min Confidence) are present
- [ ] Review button is disabled until folder selected
- [ ] Face ID list area is empty initially
- [ ] Merge controls are present
- [ ] Save button is present but disabled

### Data Loading
- [ ] Browse to participant directory (with `faces_combined.csv`)
- [ ] Review button becomes enabled
- [ ] Click "Review"
- [ ] Session filter panel appears
- [ ] Session checkboxes show all sessions
- [ ] Face ID list populates with face IDs
- [ ] Each row shows: checkbox, thumbnail, face ID, instance count
- [ ] Rows are sorted by instance count (descending)

### Session Filtering
- [ ] All sessions are selected by default
- [ ] Deselect one session
- [ ] Click "Apply Filter"
- [ ] Face ID list updates (instance counts may decrease)
- [ ] Face IDs with 0 instances after filter disappear
- [ ] Select All / Deselect All buttons work
- [ ] Try different session combinations

### Gallery View
- [ ] Double-click a face ID row
- [ ] Gallery popup opens
- [ ] Title shows "Face ID: FACE_XXXXX (N instances)"
- [ ] "Loading images..." message appears briefly
- [ ] Gallery shows grid of face crops (6 per row)
- [ ] Crops are larger than thumbnails (120x120)
- [ ] Can scroll if many instances
- [ ] Close button works
- [ ] Try face IDs with different instance counts (1, 10, 50+)

### Merging
- [ ] Select 2-3 face IDs (same person ideally)
- [ ] "Merge Selected" button becomes enabled
- [ ] Click "Merge Selected"
- [ ] Confirmation dialog appears
- [ ] After merge:
  - Merged face ID appears in list
  - Shows "merged from: ..." text
  - Instance count is sum of merged IDs
  - Merged ID has main thumbnail
  - "Unmerge" button appears

### Unmerging
- [ ] Find a merged face ID
- [ ] Click "Unmerge" button
- [ ] Confirmation dialog appears
- [ ] After unmerge:
  - Original face IDs reappear separately
  - Each has its own count
  - No more "merged from" text

### Filtering
- [ ] Set Min Instances to 5
- [ ] Click "Review"
- [ ] Only face IDs with ≥5 instances appear
- [ ] Set Min Confidence to 0.9
- [ ] Click "Review"
- [ ] List may shrink further

### Saving Results
- [ ] Perform some merges
- [ ] Click "Save Merged Results"
- [ ] Confirmation dialog appears
- [ ] After save:
  - Success message shows statistics
  - `faces_combined.backup.csv` created
  - `faces_combined.csv` updated with `merged_face_id` column
  - Original face IDs unchanged
  - Merged face IDs have new merged_face_id value
  - Unmerged faces have merged_face_id = face_id (no empty cells)

### Error Handling
- [ ] Try browsing to directory without `faces_combined.csv` (should show error)
- [ ] Try merging with only 1 selected (button stays disabled)
- [ ] Try unmerging a non-merged ID (button shouldn't appear)
- [ ] Try with session having no video file (should show placeholder thumbnails)

## Integration Tests

### Full Pipeline - Single Session
- [ ] Tab 1: Process one session
- [ ] Verify `face_detections.csv` created
- [ ] Tab 2: Assign face IDs for that participant
- [ ] Verify `faces_combined.csv` created
- [ ] Tab 3: Review and merge some face IDs
- [ ] Save results
- [ ] Verify `merged_face_id` column added

### Full Pipeline - Multiple Participants
- [ ] Tab 1: Process sessions for 2+ participants
- [ ] Tab 2: Assign face IDs for all participants
- [ ] Tab 3: Review each participant separately
- [ ] Perform merges for each
- [ ] Save results for each

### Settings Consistency
- [ ] Set project directory in Tab 1
- [ ] Switch to Tab 2
- [ ] Verify same directory is available (may need to browse again)
- [ ] Change sampling rate in Tab 1
- [ ] Process a session
- [ ] Close and reopen GUI
- [ ] Verify sampling rate is restored

### Error Recovery
- [ ] Start processing in Tab 1
- [ ] Let it run for a few seconds
- [ ] Interrupt by closing GUI (if possible)
- [ ] Reopen GUI
- [ ] Try processing again
- [ ] Verify it works

## Performance Tests

### Large Dataset
- [ ] Process session with 1000+ frames (high sampling rate)
- [ ] Verify GUI remains responsive
- [ ] Check memory usage is reasonable

### Many Face IDs
- [ ] Participant with 50+ face IDs
- [ ] Load in Tab 3
- [ ] Verify list scrolls smoothly
- [ ] Gallery view for high-instance face ID (100+)
- [ ] Verify images load (may take a few seconds)

## Edge Cases

### Empty/Missing Data
- [ ] Session with no faces detected
- [ ] Participant with no sessions
- [ ] Session with video but no faces
- [ ] Session missing video file

### Special Characters
- [ ] Participant/session names with spaces
- [ ] Participant/session names with special chars (if any)

### Concurrent Operations
- [ ] Start processing in Tab 1
- [ ] Switch to Tab 2 while Tab 1 is processing (should work)
- [ ] Try starting Tab 2 processing while Tab 1 is running (should work independently)

## Documentation Tests

### README
- [ ] Follow GUI_README.md instructions
- [ ] Verify all features described are present
- [ ] Try all workflows described
- [ ] Check troubleshooting section is helpful

### Code Quality
- [ ] No linter errors in new files
- [ ] Code is readable and well-commented
- [ ] Error messages are clear and helpful

## Sign-Off

Once all tests pass:
- [ ] All todos marked complete
- [ ] No critical bugs found
- [ ] Documentation is accurate
- [ ] Ready for production use

## Notes

Use this space to record any issues found during testing:

```
[Date] [Tab] [Issue] [Severity]
Example: 2026-01-22 Tab1 Directory tree doesn't update after browse - High

```
