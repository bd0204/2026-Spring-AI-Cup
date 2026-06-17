"""
hybrid_blend.py — NN ensemble proba × LGBM blend proba 的 hybrid 融合。

籌碼：
  NN side  = proba_ensemble_nn.npz（v38/v40/v41 三版 NN 融合，純 NN LB 0.3885，
             winner variance 已大幅降低 → seen 子集 OOF AUC 0.86）
  LGBM side= 朋友的兩版 LGBM blend（0.530+0.539 → 0.563）

最佳設定（v35 ablation 驗證 + 本輪診斷）：
  action / point：純 LGBM —— NN 這兩項遠弱於 LGBM，blend 只會拖累
  winner        ：rank-conditional —— 全程在 [0,1] rank space（免疫 calibration mismatch）
                  seen rally  → W_NN·rank(NN) + (1-W_NN)·rank(LGBM)
                  unseen rally→ 純 rank(LGBM)（維持 LGBM 0.563 排序，零下檔）
                  理由：NN winner 對 seen 選手 AUC 0.86、unseen 僅 0.73；且 NN↔LGBM
                        winner rank corr 僅 0.45（diversity 極高）→ seen blend 強力互補

用法：把 NN_PROBA、LGBM_PROBA、known_player_ids.json、test_new.csv 放在專案根目錄，
      python hybrid_blend.py → 輸出 ./submission.csv
W_NN_WINNER 可調（0.3~0.5），秒級重跑，不需重訓。
"""
import csv
import json
import numpy as np
from collections import defaultdict

# ---- 設定 ----
NN_PROBA = "./proba_ensemble_nn.npz"                 # Transformer ensemble 融合 proba（ensemble_proba.py 產出）
LGBM_PROBA = "./proba_blend_lgbm.npz"                # 隊友 LGBM repo 的 Model/proba_blend_lgbm.npz（masked0.3+unmasked0.7）
KNOWN_IDS = "./checkpoints/known_player_ids.json"    # 訓練看過的選手 ID
TEST_CSV = "./test_new.csv"
OUTPUT_PATH = "./submission.csv"

# v35 ablation 找到的最佳組合（baseline LGBM 0.531）：
#   action conditional 50/50 → +0.0049（最大增益）
#   winner rank-conditional  → +0.00006（50/50 反而 -0.0011）
#   point  純 LGBM           → ensemble 退步 -0.001~-0.0035，不融
# ⚠️ 註：v35 是 LGBM 0.531 時代；現在 LGBM blend 0.563 更強，action blend 不一定還是
#    正的（v38/v40 單版 × LGBM 0528 時 action blend 反而拖累）。但這次 NN 是穩定
#    ensemble（非單版），先照 v35 最佳組合驗證；若 action 拖累，把 ACTION_W_NN 設 0 即可。
# 實測更新：ACTION_W_NN=0.4 + W_NN_WINNER=0.4 → LB 0.5578 < LGBM baseline 0.561（退步）。
# 兇手是 action blend：LGBM 0.561 的 action 已太強（v35 +0.0049 是 0.531 時代的結論，
# 在強 LGBM 上反轉），NN ActF1~0.3 補不動只稀釋；action 又佔 LB 0.4 權重殺傷力大。
# → 退掉 action（純 LGBM），只留 winner rank-conditional 隔離其真實貢獻。
W_NN_WINNER = 0.5                # winner rank blend NN 權重（seen rally）；seen 子集 NN AUC 0.86
WINNER_RANK_CONDITIONAL = True   # unseen → 純 rank(LGBM) 零下檔
ACTION_W_NN = 0.0                # 退回純 LGBM（action blend 實測拖累）
ACTION_CONDITIONAL = False
POINT_W_NN = 0.0                 # point 純 LGBM
POINT_CONDITIONAL = False

N_ACTION, N_POINT = 19, 10


def rank01(x):
    """轉成 [0,1] 的 rank（與 inference.py _rank01 一致）。"""
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def load_known(path):
    ki = json.load(open(path))
    if isinstance(ki, list):
        seq = ki
    elif isinstance(ki, dict):
        seq = ki.get("known_player_ids", ki.get("ids", next(iter(ki.values()))))
    else:
        seq = []
    return set(int(x) for x in seq)


def build_seen(test_csv, known, uid_order):
    """rally 內所有 stroke 的雙方選手都 seen → True（與 inference.py 定義一致）。"""
    g = defaultdict(list)
    for r in csv.DictReader(open(test_csv, newline="")):
        g[int(r["rally_uid"])].append((int(r["gamePlayerId"]), int(r["gamePlayerOtherId"])))
    return np.array([
        all((p in known) and (o in known) for (p, o) in g[int(u)])
        for u in uid_order
    ])


def main():
    nn = np.load(NN_PROBA)
    lg = np.load(LGBM_PROBA)

    # 以 LGBM 的 rally_uid 為基準，把 NN 對齊過來
    nn_pos = {int(u): i for i, u in enumerate(nn["rally_uid"])}
    uid = lg["rally_uid"]
    assert all(int(u) in nn_pos for u in uid), "NN proba 缺少部分 LGBM 的 rally_uid！"
    order = [nn_pos[int(u)] for u in uid]
    nn_a, nn_p, nn_w = nn["action"][order], nn["point"][order], nn["winner"][order]
    lg_a, lg_p, lg_w = lg["action"], lg["point"], lg["winner"]
    n = len(uid)

    seen = build_seen(TEST_CSV, load_known(KNOWN_IDS), uid)  # 三任務 conditional 共用

    # ---- action / point：conditional prob-blend（seen→blend, unseen→純 LGBM）後 argmax ----
    a_blend = ACTION_W_NN * nn_a + (1.0 - ACTION_W_NN) * lg_a
    a_probs = np.where(seen[:, None], a_blend, lg_a) if ACTION_CONDITIONAL else a_blend
    final_a = a_probs.argmax(axis=1)
    p_blend = POINT_W_NN * nn_p + (1.0 - POINT_W_NN) * lg_p
    p_probs = np.where(seen[:, None], p_blend, lg_p) if POINT_CONDITIONAL else p_blend
    final_p = p_probs.argmax(axis=1)

    # ---- winner：rank-conditional ----
    nn_r, lg_r = rank01(nn_w), rank01(lg_w)
    blended = W_NN_WINNER * nn_r + (1.0 - W_NN_WINNER) * lg_r
    if WINNER_RANK_CONDITIONAL:
        final_w = np.where(seen, blended, lg_r)
        mode = (f"RANK-CONDITIONAL — seen {int(seen.sum())}/{n} → "
                f"{W_NN_WINNER:.2f}·rank(NN)+{1-W_NN_WINNER:.2f}·rank(LGBM); "
                f"unseen {n-int(seen.sum())} → 純 rank(LGBM)")
    else:
        final_w = blended
        mode = f"GLOBAL rank blend — {W_NN_WINNER:.2f}·rank(NN)+{1-W_NN_WINNER:.2f}·rank(LGBM)"

    # ---- 輸出 ----
    with open(OUTPUT_PATH, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rally_uid", "actionId", "pointId", "serverGetPoint"])
        for i in range(n):
            wr.writerow([int(uid[i]), int(final_a[i]), int(final_p[i]), float(final_w[i])])

    from collections import Counter

    def task_desc(w, cond):
        if w == 0:
            return "純 LGBM"
        base = f"NN={w}+LGBM={round(1 - w, 2)}"
        return f"CONDITIONAL {base}（seen blend, unseen 純 LGBM）" if cond else base

    print(f"=== Hybrid Blend → {OUTPUT_PATH} ({n} rallies, seen {int(seen.sum())}/{n}) ===")
    print(f"  actionId : {task_desc(ACTION_W_NN, ACTION_CONDITIONAL)}")
    print(f"  pointId  : {task_desc(POINT_W_NN, POINT_CONDITIONAL)}")
    print(f"  winner   : {mode}")
    print(f"  actionId 分佈: {dict(sorted(Counter(final_a.tolist()).items()))}")
    print(f"  pointId  分佈: {dict(sorted(Counter(final_p.tolist()).items()))}")
    print(f"  winner   : min={final_w.min():.3f} max={final_w.max():.3f} mean={final_w.mean():.3f}")


if __name__ == "__main__":
    main()
