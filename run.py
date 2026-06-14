"""
run.py — 一鍵執行完整訓練流程
  1. train_masked   → Model/lgbm_masked/proba_masked_1.npz
  2. train_unmasked → Model/lgbm_unmasked/proba_unmasked_1.npz
  3. blend          → submissions/submission_blend_lgbm_1.csv
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    ("Masked 模型訓練",   ROOT / "train" / "train_masked"   / "train_masked.py"),
    ("Unmasked 模型訓練", ROOT / "train" / "train_unmasked" / "train_unmasked.py"),
    ("Blend 融合",        ROOT / "train" / "blend"          / "blend.py"),
]

def run_step(name: str, script: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  [{name}]  {script.relative_to(ROOT)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {name} 失敗（exit code {result.returncode}），停止執行。")
        sys.exit(result.returncode)
    print(f"\n  完成：{elapsed:.1f}s")

if __name__ == "__main__":
    total_t0 = time.time()
    for name, script in STEPS:
        run_step(name, script)
    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  全流程完成，總耗時 {total/60:.1f} 分鐘")
    print(f"  結果：submissions/submission_blend_lgbm_1.csv")
    print(f"{'='*60}")
