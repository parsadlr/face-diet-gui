"""Check eye tracking synchronization."""
import csv

# Load demo_eye.tsv
gaze_data = []
sync_time = None

with open('examples/demo_eye.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    
    for row in reader:
        try:
            if row[13] == 'SyncPortOutHigh' and sync_time is None:
                sync_time = float(row[0])
                print(f"Sync event at: {sync_time}ms")
                continue
            
            if row[2] == 'Eye Tracker':
                ts = float(row[0])
                x = float(row[15])
                y = float(row[16])
                gaze_data.append((ts, x, y))
        except:
            continue

print(f"\nTotal gaze points: {len(gaze_data)}")
print(f"First gaze: {gaze_data[0][0]}ms at ({gaze_data[0][1]:.0f}, {gaze_data[0][2]:.0f})")
print(f"Last gaze: {gaze_data[-1][0]}ms at ({gaze_data[-1][1]:.0f}, {gaze_data[-1][2]:.0f})")
print(f"Duration: {(gaze_data[-1][0] - gaze_data[0][0])/1000:.2f}s")

# Check samples around 60s (1:00)
print(f"\nGaze samples around 60s mark:")
for ts, x, y in gaze_data:
    if 59900 <= ts <= 60200:
        print(f"  {ts}ms: ({x:.0f}, {y:.0f})")

# Check samples around 70s (1:10)
print(f"\nGaze samples around 70s mark:")
for ts, x, y in gaze_data:
    if 69900 <= ts <= 70200:
        print(f"  {ts}ms: ({x:.0f}, {y:.0f})")

# Calculate what video is looking for
print(f"\n--- Matching Logic Check ---")
fps = 24.95
print(f"Video FPS: {fps}")

# For frame at 1:00 in trimmed video (which is actually 1:00 in original video)
frame_60s = int(60 * fps)
print(f"\nAt 1:00 mark:")
print(f"  Frame number: {frame_60s}")
print(f"  Video time: 60.0s = 60,000ms")
print(f"  Target gaze time: {sync_time}ms (sync) + 60,000ms = {sync_time + 60000}ms")
print(f"  Looking for gaze near: {sync_time + 60000}ms")

