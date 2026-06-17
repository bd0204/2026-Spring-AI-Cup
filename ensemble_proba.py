"""
ensemble_proba.py — 多版本 NN probability-level 融合（variance reduction）。

背景：單一 NN 模型對 test 個別 rally 的預測高 variance（v38/v40/v41 三版分歧
action 38% / point 53%，winner rank 相關僅 0.60）。argmax 投票 ensemble 已驗證
LB 0.3773 → 0.3871（+0.0098）。本 script 改用 probability-level 融合，保留每個
模型的信心度，對 macro-F1（少數類不被多數決抹掉）與 winner AUC（連續機率平均）
都比 argmax 投票更優，預期再進一步。

用法：
  1. 對每個版本各跑一次 inference，設 cfg.save_nn_proba_path 存 raw NN proba：
       v38: use_point_long_view/short_expert=False, use_fp_action_prior=False,
            inference_dirs=["./ckpt_v38"],  save_nn_proba_path="./proba_v38_nn.npz"
       v40: point_*=True, fp=False, inference_dirs=["./ckpt_v40"], "./proba_v40_nn.npz"
       v41: point_*=True, fp=True,  inference_dirs=["./ckpt_v41"], "./proba_v41_nn.npz"
  2. 把 npz 路徑列在下方 PROBA_FILES
  3. python ensemble_proba.py  → 輸出 ./submission.csv

融合方式：
  action/point：raw softmax proba 平均 → prior shift（對齊 train 非發球分布，與單版
                inference 一致）→ argmax
  winner       ：sigmoid 機率平均（AUC 只看 ranking，平均直接降 variance）
"""
import csv
import os
import numpy as np

# ---- 設定（不存在的檔案會自動跳過）----
# v38/v40/v41 三個 NN 版本各自 inference 存的 raw NN proba（每個是 5-fold ensemble）
PROBA_FILES = [
    "./proba_v38_nn.npz",
    "./proba_v40_nn.npz",
    "./proba_v41_nn.npz",
]
TRAIN_PATH = "./train.csv"
OUTPUT_PATH = "./submission.csv"
# 非空 → 額外把「融合後的 raw 平均 proba」存成 npz（格式與單版 npz 完全相同：
# rally_uid/action/point/winner，都是 prior-shift 前的平均機率）。用途：
#   (a) 餵 hybrid — winner 當穩定 NN winner 跟 LGBM blend
#   (b) 階層融合 — 之後跟 multi-seed 結果再丟進本 script 一起平均
# 空字串 = 不存。
SAVE_FUSED_PROBA = "./proba_ensemble_nn.npz"
# 跨版融合方式：
#   "arithmetic" = 算術平均（簡單直觀，已驗證純 NN LB 0.3885）
#   "geometric"  = 幾何/logit 平均：action/point = softmax(mean(log p))，
#                  winner = sigmoid(mean(logit p))。懲罰分歧、獎勵共識 → 少數類更尖銳，
#                  理論上對 macro-F1 常勝算術平均（但需提交驗證，無 validation 可選）。
FUSION_MODE = "arithmetic"   # 實測 geometric 0.3867 < arithmetic 0.3885（high-variance 任務 arithmetic 較穩）
ACTION_ALPHA = 1.0   # action prior shift 強度（與單版 inference 的 action_prior_shift_alpha 一致）
POINT_ALPHA = 1.0    # point  prior shift 強度（與 pointid_prior_shift_alpha 一致）
N_ACTION, N_POINT = 19, 10


def prior_shift(p, prior, alpha):
    """prob 空間 multiplicative prior shift（等價 logit + alpha·log(prior/pred) 再 softmax）。"""
    if alpha <= 0:
        return p
    eps = 1e-9
    pred_prior = p.mean(axis=0)
    factor = ((prior + eps) / (pred_prior + eps)) ** alpha
    q = p * factor
    q /= q.sum(axis=1, keepdims=True)
    return q


def train_prior(path, col, n_classes):
    """train 非發球 (strikeNumber != 1) 的 col 分布，對齊 sliding-window target。"""
    pr = np.zeros(n_classes, dtype=np.float64)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["strikeNumber"]) != 1:
                c = int(r[col])
                if 0 <= c < n_classes:
                    pr[c] += 1.0
    return pr / max(pr.sum(), 1.0)


def load_sorted(f):
    """載入一個 npz，按 rally_uid 排序回傳，確保多版本對齊。"""
    d = np.load(f)
    uid = d["rally_uid"]
    order = np.argsort(uid)
    return uid[order], d["action"][order], d["point"][order], d["winner"][order]


def main():
    files = [f for f in PROBA_FILES if os.path.exists(f)]
    missing = [f for f in PROBA_FILES if not os.path.exists(f)]
    if missing:
        print(f"[跳過不存在] {missing}")
    if not files:
        raise SystemExit("沒有任何 proba 檔案存在！先跑 inference 存 proba。")

    eps = 1e-9
    base_uid = None
    a_sum = p_sum = w_sum = None
    for f in files:
        uid, a, p, w = load_sorted(f)
        if base_uid is None:
            base_uid = uid
            a_sum = np.zeros_like(a, dtype=np.float64)
            p_sum = np.zeros_like(p, dtype=np.float64)
            w_sum = np.zeros_like(w, dtype=np.float64)
        assert np.array_equal(uid, base_uid), f"{f} 的 rally_uid 與第一個檔案不一致！"
        if FUSION_MODE == "geometric":
            # log-space 累加：action/point 累加 log(p)；winner 累加 logit(p)
            a_sum += np.log(a + eps)
            p_sum += np.log(p + eps)
            wc = np.clip(w, 1e-7, 1.0 - 1e-7)
            w_sum += np.log(wc / (1.0 - wc))
        else:
            a_sum += a
            p_sum += p
            w_sum += w
        print(f"  + {f}  (action {a.shape}, point {p.shape}, winner {w.shape})")

    K = len(files)
    if FUSION_MODE == "geometric":
        # 幾何平均：exp(mean log) 後 re-normalize；winner = sigmoid(mean logit)
        a_avg = np.exp(a_sum / K); a_avg /= a_avg.sum(axis=1, keepdims=True)
        p_avg = np.exp(p_sum / K); p_avg /= p_avg.sum(axis=1, keepdims=True)
        w_avg = 1.0 / (1.0 + np.exp(-(w_sum / K)))
    else:
        a_avg, p_avg, w_avg = a_sum / K, p_sum / K, w_sum / K
    print(f"\n融合 {K} 個版本（{FUSION_MODE}），{len(base_uid)} rallies")

    # 額外存「融合後的 raw 平均 proba」（格式同單版 npz，供 hybrid / 階層融合用）
    if SAVE_FUSED_PROBA:
        np.savez(
            SAVE_FUSED_PROBA,
            rally_uid=base_uid,
            action=a_avg,    # (N, 19) raw 平均（prior-shift 前）
            point=p_avg,     # (N, 10)
            winner=w_avg,    # (N,)
        )
        print(f"  [fused] 融合後 raw 平均 proba 已存 → {SAVE_FUSED_PROBA}  "
              f"(action {a_avg.shape}, point {p_avg.shape}, winner {w_avg.shape})")

    # prior shift（對齊 train 分布，與單版 inference 一致）
    a_final = prior_shift(a_avg, train_prior(TRAIN_PATH, "actionId", N_ACTION), ACTION_ALPHA)
    p_final = prior_shift(p_avg, train_prior(TRAIN_PATH, "pointId", N_POINT), POINT_ALPHA)
    a_arg = a_final.argmax(axis=1)
    p_arg = p_final.argmax(axis=1)

    # 寫出 submission（按 rally_uid 排序）
    with open(OUTPUT_PATH, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rally_uid", "actionId", "pointId", "serverGetPoint"])
        for i in range(len(base_uid)):
            wr.writerow([int(base_uid[i]), int(a_arg[i]), int(p_arg[i]), float(w_avg[i])])

    # 摘要
    from collections import Counter
    print(f"\n=== Ensemble Submission ({K} versions) → {OUTPUT_PATH} ===")
    print(f"  rallies: {len(base_uid)}")
    print(f"  actionId 分佈: {dict(sorted(Counter(a_arg.tolist()).items()))}")
    print(f"  pointId  分佈: {dict(sorted(Counter(p_arg.tolist()).items()))}")
    print(f"  winner: min={w_avg.min():.3f} max={w_avg.max():.3f} mean={w_avg.mean():.3f}")


if __name__ == "__main__":
    main()
