# Retriever Pipeline Batch Processing with Timing Table

**Date:** 2026-05-10  
**Objective:** Run the retriever pipeline on all test photos and visualize per-stage timing breakdown across photos.

## Overview

Extend `02_phase2_full_pipeline.ipynb` to process all 6 test photos through the retriever pipeline, collect per-stage timing data from each run, and display results in a matplotlib table for easy comparison.

## Current State

The notebook currently has a single-photo retriever cell that:
- Runs `caption_retriever()` on only `photos[5]`
- Prints caption, match info, and per-stage timings
- Has access to timing breakdown via `r.debug.get("timings", {})`

## Design

### Phase 1: Loop Processing
- Replace single-photo retriever cell with a loop over all 6 photos in `photos` list
- For each iteration:
  1. Load image and run `caption_retriever()`
  2. Measure overall elapsed time with `time.perf_counter()`
  3. Extract per-stage timings from `r.debug["timings"]` dict
  4. Print photo results in current format (caption, match info, timings)
  5. Append timing data to a results accumulator

### Phase 2: Timing Table Visualization
- After loop completes, convert timing results to structured format (list of dicts)
- Create matplotlib figure with `ax.table()` or similar
- Table structure:
  - **Rows:** photo filenames
  - **Columns:** each stage name (from timing keys) + "Total"
  - **Values:** seconds (formatted to 2 decimal places)
  - **Styling:** optional cell coloring (lighter = faster, darker = slower)

### Stage Names
Extract from pipeline debug output. Expected stages include:
- `embedding_lookup` or similar (KB retrieval)
- `rerank` (reranking step)
- Any other stages in `caption_retriever`'s debug timings dict

### Data Structure
```python
timings_results = [
    {"photo": "temple_of_literature_1.jpg", "embedding_lookup": 0.12, "rerank": 0.05, "caption": 0.30, "total": 0.47},
    {"photo": "temple_of_literature_2.jpg", "embedding_lookup": 0.11, "rerank": 0.06, "caption": 0.31, "total": 0.48},
    ...
]
```

## Output Behavior

1. **During loop:** Print each photo's results line-by-line (existing format preserved)
2. **After loop:** Display matplotlib table showing all 6 photos × all timing stages
3. **No changes** to pipeline code or model loading

## Success Criteria

- ✓ All 6 photos processed through retriever pipeline
- ✓ Per-stage timings collected from each run
- ✓ Captions and match info printed as loop progresses
- ✓ Matplotlib table visualizes timing breakdown across all photos
- ✓ Table is readable (proper formatting, labels, units)
