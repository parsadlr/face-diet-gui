---
name: Enhance exposure time plot with error tolerance and 95% CI
overview: Modify Plot 1 (exposure time) to add error tolerance analysis, 95% CI instead of std, baseline reference lines, and convert to minutes per hour. Also update x-axis to show actual fps values at their correct positions.
todos: []
---

# Enhance Exposure Time Plot with Error Tolerance and 95% CI

## Overview

Modify the exposure time plot (Plot 1, Cell 13) to include error tolerance analysis, 95% confidence intervals, baseline reference lines, and convert exposure time to minutes per hour. Also update the x-axis to display actual FPS values at their correct positions.

## Changes Required

### 1. Configuration (Cell 1)

- Add `ERROR_TOLERANCE_PCT = 10` to the configuration section

### 2. Data Processing (Cell 5 - `process_participant` function)

- Calculate total session length in hours for each participant:
  - Count unique `(session_name, frame_number)` pairs from the full dataframe
  - Convert to hours: `total_unique_frames / ORIGINAL_FPS / 3600`
- Store `total_session_length_hours` in the results dictionary for each participant

### 3. Data Aggregation (Cell 11 - `aggregate_results` function)

- Extract `total_session_length_hours` from participant results and add it to the aggregated DataFrame
- Ensure it's available for plotting

### 4. Plot 1 Enhancement (Cell 13)

The plot currently shows mean ± std for each participant. Modify it to:

#### 4.1 Convert Exposure Time to Minutes per Hour

- For each participant's data: `exposure_time_min_per_hour = (mean / 60) / total_session_length_hours`
- Apply the same conversion to CI bounds

#### 4.2 Replace Standard Deviation with 95% CI

- Calculate 95% CI: `mean ± 1.96 * std` (instead of `mean ± std`)
- Update `fill_between` to use CI bounds

#### 4.3 Add Baseline Reference Line

- Plot a horizontal line at the baseline value (from `original_fps` entry) converted to minutes per hour
- Use a distinct style (e.g., dashed black line)

#### 4.4 Add Tolerance Lines and Shading

- Calculate tolerance bounds: `baseline ± (baseline * ERROR_TOLERANCE_PCT / 100)`
- Draw two horizontal lines for upper and lower tolerance bounds
- Shade the area between tolerance lines (e.g., light gray with low alpha)

#### 4.5 Find and Mark Lowest FPS Within Tolerance

- For each participant, iterate through configurations sorted by `target_fps` (descending)
- For each config, check if the 95% CI (`mean - 1.96*std` to `mean + 1.96*std`) is entirely within tolerance bounds
- Mark the lowest FPS (highest `target_fps`) where this condition is met
- Add a marker (e.g., vertical line or special symbol) at that point

#### 4.6 Update X-Axis to Show Actual FPS

- Instead of `downsample_factor`, use `target_fps` for x-axis values
- Use `ax.set_xscale('log')` or linear scale as appropriate
- Set x-axis labels to show actual FPS values (e.g., 25, 12, 6, 3, 1, 0.5, 0.25, etc.)
- Position labels at the actual `target_fps` positions (not equally distributed)
- Use `ax.set_xticks()` and `ax.set_xticklabels()` with the actual FPS values

### 5. Implementation Details

#### Calculate Total Session Length

```python
# In process_participant, after loading df:
unique_frames = df[['session_name', 'frame_number']].drop_duplicates()
total_unique_frames = len(unique_frames)
total_session_length_hours = total_unique_frames / ORIGINAL_FPS / 3600
# Store in results['original_fps']['total_session_length_hours']
```

#### Convert to Minutes per Hour

```python
# In plot cell:
exposure_min_per_hour = (participant_data['mean'] / 60) / participant_data['total_session_length_hours']
ci_lower = (participant_data['mean'] - 1.96 * participant_data['std']) / 60 / participant_data['total_session_length_hours']
ci_upper = (participant_data['mean'] + 1.96 * participant_data['std']) / 60 / participant_data['total_session_length_hours']
```

#### Find Lowest FPS Within Tolerance

```python
# For each participant:
baseline_min_per_hour = baseline_value / 60 / total_session_length_hours
tolerance_lower = baseline_min_per_hour * (1 - ERROR_TOLERANCE_PCT / 100)
tolerance_upper = baseline_min_per_hour * (1 + ERROR_TOLERANCE_PCT / 100)

# Sort by target_fps descending (highest to lowest)
sorted_data = participant_data.sort_values('target_fps', ascending=False)
for _, row in sorted_data.iterrows():
    ci_lower = (row['mean'] - 1.96 * row['std']) / 60 / total_session_length_hours
    ci_upper = (row['mean'] + 1.96 * row['std']) / 60 / total_session_length_hours
    if ci_lower >= tolerance_lower and ci_upper <= tolerance_upper:
        # Mark this point
        break
```

## Files to Modify

- `fps_experiment.ipynb`:
  - Cell 1: Add `ERROR_TOLERANCE_PCT` configuration
  - Cell 5: Add `total_session_length_hours` calculation in `process_participant`
  - Cell 11: Extract `total_session_length_hours` in `aggregate_results`
  - Cell 13: Complete rewrite of Plot 1 with all new features

## Notes

- The baseline value comes from the `'original_fps'` entry in results (which has `mean = baseline_exp_time`)
- X-axis should use `target_fps` from the config, not `downsample_factor`
- For `'original_fps'` entry, `target_fps = ORIGINAL_FPS` (25 fps)
- Ensure proper handling of log scale if needed for FPS values