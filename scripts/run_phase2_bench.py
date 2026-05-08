"""Three-run Phase 2 benchmark — used to validate latency optimizations."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from hanoi_caption.kb_indexer import build_or_load_index
from hanoi_caption.kb_loader import load_kb
from hanoi_caption.pipeline import caption_phase2

print("Loading KB...")
nodes = load_kb("data/kb.json")
kb_index = build_or_load_index(nodes)
print(f"KB ready: {len(nodes)} nodes")

photo = Path("tests/fixtures/temple_of_literature_1.jpg")
img = Image.open(photo).convert("RGB")

results = []
for run in range(1, 4):
    print(f"\n{'='*64}\n  Run {run}\n{'='*64}", flush=True)
    t0 = time.perf_counter()
    r = caption_phase2(image=img, kb_nodes=nodes, kb_index=kb_index)
    elapsed = time.perf_counter() - t0
    results.append((run, elapsed, r))

    print(f"\n--- Phase 2: {photo.name}  (caption generated in {elapsed:.1f}s) ---")
    print(r.caption or r.refusal)
    print(f"  regions: {r.debug.get('n_regions')}, queries: {r.debug.get('queries')}")

    timings = r.debug.get("timings", {})
    if timings:
        print("\nPer-stage timings:")
        for name, secs in timings.items():
            print(f"  {name:18s}  {secs:6.2f}s")
        print(f"  {'sum':18s}  {sum(timings.values()):6.2f}s")

    match = r.debug.get("match", {})
    top_k = match.get("top_k") or []
    if top_k:
        top1 = top_k[0]
        line = f"  match top-1: {top1['node_id']!r}  score={top1['score']:.3f}"
        if len(top_k) > 1:
            top2 = top_k[1]
            line += f"   top-2: {top2['node_id']!r}  score={top2['score']:.3f}   gap={top1['score']-top2['score']:.3f}"
        print(line, flush=True)

print(f"\n{'='*64}\n  Summary\n{'='*64}")
for run, elapsed, _ in results:
    print(f"  Run {run}: {elapsed:6.1f}s")
