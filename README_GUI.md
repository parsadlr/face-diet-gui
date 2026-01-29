# Face-Diet GUI Documentation

Welcome to the Face-Diet Multi-Tab GUI!

## 🚀 Quick Start

```bash
# Install GUI dependency
pip install customtkinter

# Run the GUI
python run_gui.py
```

That's it! The GUI will open with 3 tabs for the complete face processing pipeline.

## 📚 Documentation

Choose the guide that fits your needs:

### For New Users
👉 **[GUI_README.md](GUI_README.md)** - Complete user guide
- How to use each tab
- Settings explained
- Common workflows
- Troubleshooting

### For Existing Users
👉 **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Transition from old GUI
- What's changed
- Feature comparison
- File compatibility
- Migration steps

### For Testing
👉 **[TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)** - Comprehensive test plan
- Feature verification
- Integration tests
- Edge cases
- Performance tests

### For Developers
👉 **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- Architecture overview
- Implementation details
- File structure
- Future enhancements

## 🎯 What's New

### Three Tabs, One Workflow

1. **Video Processing** (Tab 1)
   - Process videos to detect faces
   - Extract demographic attributes
   - Select specific sessions to process

2. **Face ID Assignment** (Tab 2)
   - Cluster faces across sessions
   - Adjust clustering parameters
   - See results in real-time

3. **Manual Review & Merging** (Tab 3)
   - Review face IDs with thumbnails
   - **NEW**: Filter by session
   - **NEW**: Gallery view (double-click)
   - Merge incorrect clusters
   - Save results

### Key Features

✨ **All-in-One Interface** - Complete pipeline in one window
✨ **Session Filtering** - Focus on specific sessions in review
✨ **Gallery View** - See all instances of a face ID
✨ **Settings Persistence** - Your settings are saved automatically
✨ **Progress Tracking** - Clear status updates and detailed logs
✨ **Backward Compatible** - Works with existing data

## 📁 File Structure

```
Your Project/
├── Participant1/
│   ├── Session1/
│   │   ├── scenevideo.mp4
│   │   ├── eye_tracking.tsv
│   │   └── face_detections.csv     ← Created by Tab 1
│   ├── Session2/
│   │   └── ...
│   └── faces_combined.csv          ← Created by Tab 2
├── Participant2/
│   └── ...
```

## 🎓 Learning Path

### First Time User?

1. Start with [GUI_README.md](GUI_README.md)
2. Follow the "Tab 1" section to process your first session
3. Continue through Tab 2 and Tab 3
4. Refer to troubleshooting as needed

### Experienced User?

1. Skim [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for what's new
2. Jump directly to the tab you need
3. Try the new features (session filtering, gallery view)

### Developer?

1. Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. Examine the code structure
3. Check [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) for validation

## 💡 Common Tasks

### Process New Sessions
- Open Tab 1
- Select sessions
- Click "Start Processing"

### Assign Face IDs
- Open Tab 2
- Select participants
- Click "Assign Face IDs"

### Review and Merge
- Open Tab 3
- Browse to participant
- Click "Review"
- Double-click faces to view gallery
- Merge as needed
- Save results

## ❓ Need Help?

### Troubleshooting

**"No files found"**
→ Check [GUI_README.md § Troubleshooting](GUI_README.md#troubleshooting)

**"How do I migrate my data?"**
→ See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

**"What settings should I use?"**
→ Check [GUI_README.md § Settings Explained](GUI_README.md)

**"Gallery view not working"**
→ Verify video files are present in session directories

### Still Stuck?

1. Check terminal output for error messages
2. Verify file structure matches documentation
3. Try with a small test dataset first
4. Review relevant documentation section

## 🔧 System Requirements

- Python 3.10+
- customtkinter
- All dependencies from requirements.txt
- OpenCV for video processing
- FAISS for clustering (Tab 2)

## 📝 Version Info

**Current Version**: 1.0
**Release Date**: January 22, 2026
**Status**: Ready for Testing

## 🗺️ Documentation Map

```
README_GUI.md (You are here!)
├── GUI_README.md ...................... Complete user guide
├── MIGRATION_GUIDE.md ................. For existing users
├── TESTING_CHECKLIST.md ............... Test procedures
└── IMPLEMENTATION_SUMMARY.md .......... Technical details
```

## 🚦 Next Steps

1. **Install**: `pip install customtkinter`
2. **Run**: `python run_gui.py`
3. **Learn**: Read [GUI_README.md](GUI_README.md)
4. **Explore**: Try each tab with sample data
5. **Enjoy**: Process your real datasets!

---

**Happy Face Processing! 🎭**

For detailed instructions, please refer to the specific documentation files listed above.
