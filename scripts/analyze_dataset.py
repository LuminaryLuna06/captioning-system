import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('D:/Jupiter/captioning-system/data/dataset_Hanoi Tourism Dataset.json', encoding='utf-8') as f:
    d = json.load(f)

videos = d['videos']

n_annotated_segs = 0
n_total_segs = 0
n_videos_with_annotated = 0
all_landmark_names = set()

for v in videos:
    segs = v.get('segments', [])
    annotated = 0
    for s in segs:
        n_total_segs += 1
        regions = s.get('regions', [])
        has_cap = any(r.get('captions', {}).get('en', {}).get('combined') for r in regions)
        if has_cap:
            n_annotated_segs += 1
            annotated += 1
            all_landmark_names.add(s.get('name', ''))
    if annotated > 0:
        n_videos_with_annotated += 1

with_segs = sum(1 for v in videos if v.get('segments'))
print(f'Videos with segments: {with_segs}')
print(f'Total segments: {n_total_segs}')
print(f'Segments with annotated captions: {n_annotated_segs}')
print(f'Videos with at least 1 annotated segment: {n_videos_with_annotated}')
print(f'Unique landmark names: {len(all_landmark_names)}')
print()
print('Landmark names found:')
for n in sorted(all_landmark_names):
    print(' ', n)
