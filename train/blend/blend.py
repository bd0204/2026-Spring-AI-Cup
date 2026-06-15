"""
Blend two .npz probability files and output submission.csv.

Three toggle switches control which tasks are blended:
  BLEND_ACTION  - actionId   (19 classes, effective 0-14)
  BLEND_POINT   - pointId    (10 classes, 0-9)
  BLEND_WINNER  - serverGetPoint (continuous probability)

When a switch is False, the task result is taken directly from FALLBACK (either "A" or "B").
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 路徑設定
ROOT_DIR = Path(__file__).parent.parent.parent

# ─── Configuration ────────────────────────────────────────────────────────────

FILE_A = ROOT_DIR / "Model" / "lgbm_masked" / "proba_masked.npz"   # first model
FILE_B = ROOT_DIR / "Model" / "lgbm_unmasked" / "proba_unmasked.npz"     # second model

WEIGHT_A = 0.30   # weight for FILE_A  (must sum to 1 with WEIGHT_B)
WEIGHT_B = 0.70   # weight for FILE_B

# Toggle: True = blend the task, False = use FALLBACK directly
BLEND_ACTION = True
BLEND_POINT  = True
BLEND_WINNER = True

# When a task switch is False, use this file's prediction: "A" or "B"
FALLBACK = "B"

OUTPUT_CSV  = ROOT_DIR / "submissions" / "submission_blend_lgbm.csv"
OUTPUT_NPZ  = ROOT_DIR / "Model" / "proba_blend_lgbm.npz"

# ─── Load ─────────────────────────────────────────────────────────────────────

a = np.load(FILE_A)
b = np.load(FILE_B)

assert np.array_equal(a["rally_uid"], b["rally_uid"]), "rally_uid mismatch between files"
rally_uid = a["rally_uid"]

assert FALLBACK in ("A", "B"), 'FALLBACK must be "A" or "B"'
fb = a if FALLBACK == "A" else b

# ─── Blend function ───────────────────────────────────────────────────────────

def blend_proba(pa: np.ndarray, pb: np.ndarray, wa: float, wb: float) -> np.ndarray:
    """Weighted average of two probability arrays (row-wise normalization for matrix)."""
    mixed = wa * pa + wb * pb
    if mixed.ndim == 2:
        row_sums = mixed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        mixed = mixed / row_sums
    return mixed

# ─── Action ───────────────────────────────────────────────────────────────────

if BLEND_ACTION:
    action_proba = blend_proba(a["action"], b["action"], WEIGHT_A, WEIGHT_B)
    action_pred  = np.argmax(action_proba, axis=1)
else:
    action_pred = np.argmax(fb["action"], axis=1)

# ─── Point ────────────────────────────────────────────────────────────────────

if BLEND_POINT:
    point_proba = blend_proba(a["point"], b["point"], WEIGHT_A, WEIGHT_B)
    point_pred  = np.argmax(point_proba, axis=1)
else:
    point_pred = np.argmax(fb["point"], axis=1)

# ─── Winner ───────────────────────────────────────────────────────────────────

if BLEND_WINNER:
    winner_proba = blend_proba(a["winner"], b["winner"], WEIGHT_A, WEIGHT_B)
else:
    winner_proba = fb["winner"]

# ─── Assemble submission ───────────────────────────────────────────────────────

submission = pd.DataFrame({
    "rally_uid":      rally_uid,
    "actionId":       action_pred.astype(int),
    "pointId":        point_pred.astype(int),
    "serverGetPoint": winner_proba,
})

submission.to_csv(OUTPUT_CSV, index=False)
print(f"Saved → {OUTPUT_CSV}  ({len(submission)} rows)\n")

# ─── Save blended probabilities ───────────────────────────────────────────────

if BLEND_ACTION:
    save_action = action_proba
else:
    save_action = fb["action"] if fb["action"].ndim == 2 else fb["action"]

if BLEND_POINT:
    save_point = point_proba
else:
    save_point = fb["point"] if fb["point"].ndim == 2 else fb["point"]

np.savez(
    OUTPUT_NPZ,
    rally_uid=rally_uid,
    action=save_action,
    point=save_point,
    winner=winner_proba,
)
print(f"Saved → {OUTPUT_NPZ}\n")

# ─── Statistics ───────────────────────────────────────────────────────────────

print("=== actionId distribution (0-14) ===")
action_counts = submission["actionId"].value_counts().sort_index()
for cls, cnt in action_counts.items():
    print(f"  class {cls:2d}: {cnt}")
print(f"  total : {action_counts.sum()}\n")

print("=== pointId distribution (0-9) ===")
point_counts = submission["pointId"].value_counts().sort_index()
for cls, cnt in point_counts.items():
    print(f"  class {cls}: {cnt}")
print(f"  total : {point_counts.sum()}\n")

print("=== serverGetPoint (winner) statistics ===")
w = submission["serverGetPoint"]
pred_label = (w >= 0.5).astype(int)
label_counts = pred_label.value_counts().sort_index()
print(f"  predicted 0 (receiver wins): {label_counts.get(0, 0)}")
print(f"  predicted 1 (server  wins ): {label_counts.get(1, 0)}")
print(f"  mean={w.mean():.4f}  std={w.std():.4f}  min={w.min():.4f}  max={w.max():.4f}")

# ─── Mode summary ─────────────────────────────────────────────────────────────

print("\n=== Blend mode ===")
fallback_label = f"FALLBACK({FALLBACK}={FILE_A if FALLBACK == 'A' else FILE_B})"
print(f"  action : {'BLENDED' if BLEND_ACTION else fallback_label}")
print(f"  point  : {'BLENDED' if BLEND_POINT  else fallback_label}")
print(f"  winner : {'BLENDED' if BLEND_WINNER else fallback_label}")
