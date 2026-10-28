import pandas as pd
import numpy as np

gt = pd.read_csv('C:/Users/KALYAN/.gemini/antigravity/scratch/Vision_Object_Tracking/results/stage8/ground_truth.csv').set_index('frame')

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0

trackers = [
    ('Stage 5 (Baseline)', 'results/stage5_tracking_data.csv'),
    ('Stage 6A (Original)', 'results/stage6A_tracking_data.csv'),
    ('Stage 6A (Corrected)', 'results/stage6A_corrected_tracking_data.csv'),
    ('Stage 9A (Transition)', 'results/stage9A_transition_aware_data.csv'),
    ('Stage 9C (Appearance)', 'results/stage9C_appearance_aware_data.csv'),
    ('Stage 10 (Master)', 'results/stage10_master_tracking_data.csv')
]

base = 'C:/Users/KALYAN/.gemini/antigravity/scratch/Vision_Object_Tracking/'
print('='*92)
print(f"{'Tracker Stage':24} | {'Mean IoU':>9} | {'IoU >= 0.50':>12} | {'Center Err':>11} | {'Late IoU':>9}")
print('='*92)
for name, rel in trackers:
    df = pd.read_csv(base + rel).set_index('frame')
    common = sorted(list(set(gt.index) & set(df.index)))
    ious, errs = [], []
    for f in common:
        g, p = gt.loc[f], df.loc[f]
        gb = (g['gt_x'], g['gt_y'], g['gt_width'], g['gt_height'])
        pb = (p['x'], p['y'], p['width'], p['height'])
        ious.append(iou(gb, pb))
        errs.append(np.hypot(gb[0]+gb[2]/2 - (pb[0]+pb[2]/2), gb[1]+gb[3]/2 - (pb[1]+pb[3]/2)))
    late_ious = [ious[common.index(f)] for f in common if f >= 260]
    print(f"{name:24} | {np.mean(ious):9.3f} | {np.mean(np.array(ious)>=0.5)*100:11.1f}% | {np.mean(errs):9.1f}px | {np.mean(late_ious):9.3f}")
print('='*92)
