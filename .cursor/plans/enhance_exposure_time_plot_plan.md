# Enhance Exposure Time Plot with Error Tolerance and 95% CI

## Overview
Modify Plot 1 (exposure time plot in Cell 13) to add error tolerance analysis, 95% confidence intervals, baseline reference lines, and convert exposure time to minutes per hour. Also update the x-axis to display actual FPS values at their correct positions.

## Changes Required

### 1. Configuration (Cell 1)
- Add `ERROR_TOLERANCE_PCT = 10` to the configuration section after `MIN_FACE_ID_COUNT`

### 2. Data Processing (Cell 5 - `process_participant` function)
- After calculating `baseline_exp_time`, calculate total session length in hours:
  ```python
  # Count unique (session_name, frame_number) pairs
  unique_frames = df[['session_name', 'frame_number']].drop_duplicates()
  total_unique_frames = len(unique_frames)
  total_session_length_hours = total_unique_frames / ORIGINAL_FPS / 3600
  ```
- Store `total_session_length_hours` in `results['original_fps']` dictionary so it's available for all configs

### 3. Data Aggregation (Cell 11 - `aggregate_results` function)
- Extract `total_session_length_hours` from the `'original_fps'` entry in each participant's results
- Add `total_session_length_hours` column to the aggregated DataFrame for each participant

### 4. Plot 1 Complete Rewrite (Cell 13)
Replace the entire plot cell with enhanced version:

#### 4.1 Data Preparation
- For each participant, get their `total_session_length_hours` from the dataframe
- Convert exposure time values to minutes per hour:
  - `exposure_min_per_hour = (mean / 60) / total_session_length_hours`
  - `ci_lower = (mean - 1.96 * std) / 60 / total_session_length_hours`
  - `ci_upper = (mean + 1.96 * std) / 60 / total_session_length_hours`
- Get baseline value from `'original_fps'` entry and convert to minutes per hour
- Calculate tolerance bounds: `baseline_min_per_hour * (1 ± ERROR_TOLERANCE_PCT/100)`

#### 4.2 Plot Structure
- Keep two subplots (All Faces and Attended Faces Only)
- For each participant:
  - Plot mean exposure time (in min/hour) vs `target_fps` (not `downsample_factor`)
  - Use `fill_between` with 95% CI bounds (mean ± 1.96*std) instead of ±std
  - Find lowest FPS where 95% CI stays within tolerance:
    - Sort by `target_fps` descending (highest to lowest)
    - Check if `ci_lower >= tolerance_lower` and `ci_upper <= tolerance_upper`
    - Mark this point with a special marker (e.g., vertical line, star, or different color)

#### 4.3 Reference Lines
- Add horizontal line for baseline (true value from 25fps) - use dashed black line
- Add two horizontal lines for tolerance bounds (upper and lower) - use dashed red lines or similar
- Shade the area between tolerance bounds with light color (e.g., light gray with alpha=0.1)

#### 4.4 X-Axis Configuration
- Use `target_fps` for x-axis values instead of `downsample_factor`
- Get unique `target_fps` values from the dataframe, sorted
- Set x-axis ticks to these actual FPS positions: `ax.set_xticks(unique_fps_values)`
- Set x-axis labels to show FPS values: `ax.set_xticklabels(unique_fps_values)`
- Use log scale if appropriate: `ax.set_xscale('log')` or linear scale
- X-axis label: "Frame Rate (fps)" instead of "Downsampling Factor"

#### 4.5 Y-Axis Configuration
- Y-axis label: "Exposure Time (minutes per hour)" instead of "Exposure Time (seconds)"

### 5. Implementation Details

#### Calculate Total Session Length
```python
# In process_participant, after loading df and before processing configs:
unique_frames = df[['session_name', 'frame_number']].drop_duplicates()
total_unique_frames = len(unique_frames)
total_session_length_hours = total_unique_frames / ORIGINAL_FPS / 3600

# Store in results['original_fps']:
results['original_fps'] = {
    ...
    'total_session_length_hours': total_session_length_hours
}
```

#### Find Lowest FPS Within Tolerance
```python
# For each participant:
baseline_row = participant_data[participant_data['config_name'] == 'original_fps'].iloc[0]
baseline_min_per_hour = (baseline_row['mean'] / 60) / baseline_row['total_session_length_hours']
tolerance_lower = baseline_min_per_hour * (1 - ERROR_TOLERANCE_PCT / 100)
tolerance_upper = baseline_min_per_hour * (1 + ERROR_TOLERANCE_PCT / 100)

# Sort by target_fps descending (highest to lowest FPS)
sorted_data = participant_data.sort_values('target_fps', ascending=False)
lowest_fps_within_tolerance = None

for _, row in sorted_data.iterrows():
    if row['config_name'] == 'original_fps':
        continue  # Skip baseline
    mean_min_per_hour = (row['mean'] / 60) / row['total_session_length_hours']
    ci_lower = (row['mean'] - 1.96 * row['std']) / 60 / row['total_session_length_hours']
    ci_upper = (row['mean'] + 1.96 * row['std']) / 60 / row['total_session_length_hours']
    
    if ci_lower >= tolerance_lower and ci_upper <= tolerance_upper:
        lowest_fps_within_tolerance = row['target_fps']
        # Mark this point on the plot
        ax.axvline(x=row['target_fps'], color=colors[i], linestyle='--', alpha=0.7, linewidth=2)
        break
```

#### X-Axis Setup
```python
# Get all unique target_fps values, sorted
all_fps_values = sorted(df_all['target_fps'].unique(), reverse=True)  # Highest to lowest

# Set ticks and labels at actual FPS positions
ax.set_xticks(all_fps_values)
ax.set_xticklabels([f"{fps:.2f}" if fps < 1 else f"{int(fps)}" for fps in all_fps_values])
ax.set_xscale('log')  # or 'linear' depending on what looks better
```

## Files to Modify
- `fps_experiment.ipynb`:
  - **Cell 1**: Add `ERROR_TOLERANCE_PCT = 10` configuration
  - **Cell 5**: Add `total_session_length_hours` calculation in `process_participant`
  - **Cell 11**: Extract and include `total_session_length_hours` in `aggregate_results`
  - **Cell 13**: Complete rewrite of Plot 1 with all new features

## Notes
- The baseline value comes from the `'original_fps'` entry (which has `mean = baseline_exp_time`)
- For `'original_fps'`, `target_fps = ORIGINAL_FPS` (25 fps)
- The 95% CI uses 1.96 as the multiplier (standard for 95% confidence)
- Tolerance is calculated as a percentage of the baseline value
- X-axis labels should be positioned at the actual FPS values, not equally spaced
- The tolerance check ensures the entire 95% CI is within the tolerance bounds (not just the mean)

