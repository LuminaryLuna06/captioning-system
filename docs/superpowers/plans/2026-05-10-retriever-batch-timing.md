# Retriever Batch Processing with Timing Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process all 6 test photos through the retriever pipeline, collect per-stage timing data, and display results in a matplotlib table for comparison.

**Architecture:** Replace the existing single-photo retriever cell with a loop that processes all photos sequentially, collecting timing breakdowns. After the loop completes, create a new cell that visualizes all timing data in a matplotlib table with rows for each photo and columns for each timing stage.

**Tech Stack:** Python (time module, matplotlib), existing pipeline API (caption_retriever)

---

## Task 1: Replace Single-Photo Retriever Cell with Batch Loop

**Files:**
- Modify: `notebooks/02_phase2_full_pipeline.ipynb` — cell id "e7f42ce0" (retriever-only section)

**Context:**
The current cell (id "e7f42ce0") runs `caption_retriever()` on only `photos[5]`. You will replace this with a loop over all 6 photos that collects timing data.

- [ ] **Step 1: Replace cell content with batch processing loop**

Replace the cell content (id "e7f42ce0") with:

```python
import time
from hanoi_caption.pipeline_retriever import caption_retriever

retriever_results = []
timing_data = []  # List of dicts: {photo, stage1, stage2, ..., total}

print("=" * 80)
print("RETRIEVER PIPELINE: BATCH PROCESSING ALL PHOTOS")
print("=" * 80)

for photo_idx, p in enumerate(photos, start=1):
    img = Image.open(p).convert("RGB")
    
    t0 = time.perf_counter()
    r = caption_retriever(image=img, kb_nodes=nodes, kb_index=kb_index)
    elapsed = time.perf_counter() - t0
    
    retriever_results.append(r)
    
    # Build timing dict for this photo
    photo_timings = {"photo": p.name, "total": elapsed}
    stage_timings = r.debug.get("timings", {})
    photo_timings.update(stage_timings)
    timing_data.append(photo_timings)
    
    # Print per-photo results (same format as original cell)
    print(f"\n--- Retriever-only [{photo_idx}/6]: {p.name}  (caption generated in {elapsed:.1f}s) ---")
    print(r.caption or r.refusal)
    
    match = r.debug.get("match", {})
    chosen_id = match.get("node_id")
    top_k = match.get("top_k") or []
    
    def _name(node_id):
        return nodes[node_id].name_en if node_id in nodes else "?"
    
    if chosen_id and chosen_id in nodes:
        print(f"\nMatched: {_name(chosen_id)!r}  (id {chosen_id}, confidence {match.get('confidence', 0):.3f})")
    elif top_k:
        print(f"\nNo match (rerank refused). Cosine top-1 was {_name(top_k[0]['node_id'])!r} @ {top_k[0]['score']:.3f}")
    
    if top_k:
        print("\nCosine top-5:")
        for c in top_k[:5]:
            marker = "  <-- rerank pick" if c["node_id"] == chosen_id else ""
            print(f"  {c['score']:.3f}  {_name(c['node_id'])!r}{marker}")
    
    if stage_timings:
        print("\nPer-stage timings:")
        for name, secs in stage_timings.items():
            print(f"  {name:14s}  {secs:6.2f}s")
        print(f"  {'total':14s}  {sum(stage_timings.values()):6.2f}s")

print("\n" + "=" * 80)
print(f"Completed batch processing of {len(retriever_results)} photos")
print("=" * 80)
```

- [ ] **Step 2: Run cell to verify batch processing works**

Execute the cell and verify:
- All 6 photos are processed (you see output for each one: `[1/6]`, `[2/6]`, ..., `[6/6]`)
- Each photo prints caption, match info, and per-stage timings
- No errors occur during processing
- `timing_data` list is populated (will see it in the variable list)

Expected output: 6 photo blocks with headers like `--- Retriever-only [1/6]: temple_of_literature_1.jpg ---`

- [ ] **Step 3: Commit the modified cell**

```bash
git add notebooks/02_phase2_full_pipeline.ipynb
git commit -m "feat(notebook): batch retriever pipeline processing for all 6 photos"
```

---

## Task 2: Create Matplotlib Timing Comparison Table

**Files:**
- Modify: `notebooks/02_phase2_full_pipeline.ipynb` — add new cell after cell id "e7f42ce0"

**Context:**
After the batch loop completes, you will add a new cell that visualizes all timing data in a matplotlib table. The table will have:
- **Rows:** one per photo (6 rows), showing filename
- **Columns:** stage names (extracted from timing_data) + "total" column
- **Values:** seconds formatted to 2 decimal places
- **Styling:** alternating row colors for readability, left-aligned text

- [ ] **Step 1: Create timing table visualization cell**

Add a new cell immediately after the batch processing cell with content:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Extract timing stage names (same across all photos, from first result's keys)
if timing_data:
    # Get all unique stage names (exclude 'photo' and 'total')
    stage_names = sorted(set(key for td in timing_data for key in td.keys() if key not in ['photo', 'total']))
    
    # Build table data
    table_rows = []
    for td in timing_data:
        row = [td['photo']]
        for stage in stage_names:
            row.append(f"{td.get(stage, 0):.2f}s")
        row.append(f"{td.get('total', 0):.1f}s")
        table_rows.append(row)
    
    # Column headers
    columns = ['Photo'] + [s.replace('_', '\n') for s in stage_names] + ['Total']
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25] + [0.1] * len(stage_names) + [0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows with alternating colors
    for i in range(1, len(table_rows) + 1):
        bg_color = '#f0f0f0' if i % 2 == 0 else '#ffffff'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(bg_color)
            # Highlight total column
            if j == len(columns) - 1:
                table[(i, j)].set_facecolor('#fffacd')
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Retriever Pipeline: Per-Stage Timing Breakdown (All 6 Photos)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nTiming Summary Statistics:")
    print("-" * 60)
    for stage in ['total'] + stage_names:
        values = [td.get(stage, 0) for td in timing_data]
        if values:
            print(f"  {stage:20s}  min: {min(values):6.3f}s  max: {max(values):6.3f}s  avg: {sum(values)/len(values):6.3f}s")
else:
    print("No timing data collected. Make sure the batch processing cell ran successfully.")
```

- [ ] **Step 2: Run cell to display the timing table**

Execute the cell and verify:
- A matplotlib table appears with 6 rows (one per photo) and columns for each timing stage plus total
- Header row is green with white text
- Data rows have alternating white and light gray backgrounds
- Total column is highlighted in yellow/light yellow
- All timing values are formatted to 2-3 decimal places
- Summary statistics print below the table showing min/max/avg for each stage

Expected output:
- Visual table with all 6 photos and their timing breakdowns
- Text summary with min/max/avg statistics for each timing stage

- [ ] **Step 3: Commit the new visualization cell**

```bash
git add notebooks/02_phase2_full_pipeline.ipynb
git commit -m "feat(notebook): add matplotlib timing comparison table for batch retriever results"
```

---

## Summary

**Total implementation time:** ~10 minutes

**Changes:**
1. Replace single-photo retriever cell with a loop processing all 6 photos
2. Add new matplotlib table cell for timing visualization
3. Both preserve existing debug output (captions, match info) for transparency

**Next:**
- Run the notebook end-to-end
- Compare timing across all 6 photos using the visualization table
