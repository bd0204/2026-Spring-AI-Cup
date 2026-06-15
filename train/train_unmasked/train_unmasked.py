import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(32)

ROOT_DIR       = Path(__file__).parent.parent.parent
SUBMISSION_DIR = ROOT_DIR / "submissions"
MODEL_DIR      = ROOT_DIR / "Model" / "lgbm_unmasked"
DATA_DIR       = ROOT_DIR / "data"
TEST_DIR       = ROOT_DIR / "test"
os.makedirs(SUBMISSION_DIR, exist_ok=True)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

# ══════════════════════════════════════════════════════════════════
# 常數
# ══════════════════════════════════════════════════════════════════
# 選手 ID 遮蔽改為依 test 決定：test 出現的 ID 在 train 保留，其餘歸零（見下方載入後處理）
MATCH_ZERO_PROB     = 0   # 40->60訓練時隨機將 match 設為 0
N_FOLDS             = 5
SHORT_RALLY_BOOST_1 = 1.5    # strikeNumber=1（預測第2拍）加權倍率，對齊 test 27.5% 比例
SHORT_RALLY_BOOST_2 = 1.3    # strikeNumber=2（預測第3拍）加權倍率，對齊 test 25.7% 比例

ACTION_CATEGORY = {0: 0}
ACTION_CATEGORY.update({i: 1 for i in range(1,  8)})   # attack
ACTION_CATEGORY.update({i: 2 for i in range(8,  12)})  # control
ACTION_CATEGORY.update({i: 3 for i in range(12, 15)})  # defensive
ACTION_CATEGORY.update({i: 4 for i in range(15, 19)})  # serve

def pt_x(pid_series):
    """pointId → 正反手座標 (1=正手 2=中 3=反手)"""
    x = ((pid_series - 1) % 3 + 1).where(pid_series > 0, 0)
    return x.astype(float)

def pt_y(pid_series):
    """pointId → 深淺座標 (1=短 2=半出台 3=長)"""
    y = ((pid_series - 1) // 3 + 1).where(pid_series > 0, 0)
    return y.astype(float)

def cross_court_indicator(hand_series, px_series):
    """
    正手打到反手側 or 反手打到正手側 → 斜線(1)
    正手打到正手側 or 反手打到反手側 → 直線(0)
    其他（含中路 / 無資料）→ -1
    """
    cross    = ((hand_series == 1) & (px_series == 3)) | \
               ((hand_series == 2) & (px_series == 1))
    straight = ((hand_series == 1) & (px_series == 1)) | \
               ((hand_series == 2) & (px_series == 3))
    result = pd.Series(-1, index=hand_series.index)
    result[cross]    = 1
    result[straight] = 0
    return result.astype(int)

# ══════════════════════════════════════════════════════════════════
#   讀取資料並合併兩份訓練集
#
#   train.csv            : 主要訓練資料（比賽 ID 1~321，選手 ID 1~196）
#   processed_train_e.csv: 額外訓練資料（比賽 ID 1~196，選手 ID 1~165）
#
#   兩份資料的 rally_uid / match / gamePlayerId 均從 1 開始，
#   直接合併會造成 ID 衝突（相同 ID 指不同對象）。
#   解決方式：對 processed 的所有 ID 加上足夠大的偏移量，
#   使兩份資料的 ID 空間完全不重疊。
#
#   偏移量設計（取整數倍，方便辨識）：
#     rally_uid       : +20000  (train max=15187, proc max=15431)
#     match           : +500    (train max=321,   proc max=196)
#     gamePlayerId    : +300    (train max=196,   proc max=144)
#     gamePlayerOtherId: +300
# ══════════════════════════════════════════════════════════════════
PROC_RALLY_OFFSET  = 20000
PROC_MATCH_OFFSET  = 500
PROC_PLAYER_OFFSET = 300

# train_e.csv 偏移量（ID 空間與前兩份完全分開）
PROC2_RALLY_OFFSET  = 60000
PROC2_MATCH_OFFSET  = 1500
PROC2_PLAYER_OFFSET = 300

# train_k.csv 偏移量（去重後的唯一 rows，ID 空間再分開）
PROC3_RALLY_OFFSET  = 80000
PROC3_MATCH_OFFSET  = 2000

def _raw_to_train(df, extra_drop_cols, rally_offset, match_offset):
    """train_e / train_k 原始格式 → 與 train.csv 相容的格式"""
    df = df.copy()
    df = df.rename(columns={"strickNumber": "strikeNumber", "strickId": "strikeId"})
    drop = ["rally_id", "let","serverGetPoint","serveId","serveNumber"] + extra_drop_cols   # 保留 serveId, serveNumber 供特徵使用
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df["_no_server_lbl"] = True
    df["rally_uid"] += rally_offset
    df["match"]     += match_offset
    df["gamePlayerId"]      = 0
    df["gamePlayerOtherId"] = 0
    return df

if __name__ == "__main__":
    _log_path = ROOT_DIR / "train" / "train_unmasked" / "train_log.txt"
    _log_file = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    print("載入資料...")
    train_main  = pd.read_csv(DATA_DIR / "train.csv")
    train_proc  = pd.read_csv(DATA_DIR / "processed_train_e.csv")
    train_k_raw = pd.read_csv(DATA_DIR / "train_k.csv")
    test_raw    = pd.read_csv(TEST_DIR / "test_new.csv")
    train_e_raw = pd.read_csv(DATA_DIR / "train_e.csv")

    # processed_train_e 的 ID 加偏移，避免與 train.csv 衝突
    train_proc = train_proc.copy()
    train_proc["rally_uid"]         += PROC_RALLY_OFFSET
    train_proc["match"]             += PROC_MATCH_OFFSET
    train_proc["gamePlayerId"]       = 0
    train_proc["gamePlayerOtherId"]  = 0

    # ── train_e.csv：欄位整理 + 偏移 ──────────────────────────────
    train_e = _raw_to_train(train_e_raw, [], PROC2_RALLY_OFFSET, PROC2_MATCH_OFFSET)

    # ── train_k.csv：先去重再過濾 ─────────────────────────────────
    # 去重：移除 train_k 中與 train_e 完全相同的 rows（共同欄位比對）
    _common = [c for c in train_k_raw.columns if c in train_e_raw.columns]
    _e_tuples = set(map(tuple, train_e_raw[_common].values))
    _k_in_e   = train_k_raw[_common].apply(lambda r: tuple(r) in _e_tuples, axis=1)
    train_k_dedup = train_k_raw[~_k_in_e].copy()
    # 只保留 >1 拍的 rally
    _k_sizes  = train_k_dedup.groupby("rally_uid").size()
    _valid_k  = _k_sizes[_k_sizes > 1].index
    train_k_dedup = train_k_dedup[train_k_dedup["rally_uid"].isin(_valid_k)]
    train_k = _raw_to_train(train_k_dedup, [], PROC3_RALLY_OFFSET, PROC3_MATCH_OFFSET)
    print(f"  train_k 去重+過濾   : {len(train_k_dedup)} rows → {train_k.shape} (rally>1拍: {len(_valid_k)})")

    # 「已見選手」在歸零前從原始 train.csv 取，確保不因歸零而遺漏選手
    train_players = (set(train_main["gamePlayerId"].unique()) |
                     set(train_main["gamePlayerOtherId"].unique())) - {0}

    # test 中出現的選手 ID → 在 train.csv 中保留原值；其餘（test 未出現）歸零
    # 邏輯：若雙方其中一方 test 沒出現，只把那一方設為 0，另一方若 test 有出現則保留
    test_player_ids = (set(test_raw["gamePlayerId"].unique()) |
                       set(test_raw["gamePlayerOtherId"].unique())) - {0}
    train_main = train_main.copy()
    train_main.loc[~train_main["gamePlayerId"].isin(test_player_ids),      "gamePlayerId"]      = 0
    train_main.loc[~train_main["gamePlayerOtherId"].isin(test_player_ids), "gamePlayerOtherId"] = 0

    print(f"  test 中出現的選手數  : {len(test_player_ids)}")
    _kept_p = (train_main["gamePlayerId"] != 0).sum()
    _kept_o = (train_main["gamePlayerOtherId"] != 0).sum()
    print(f"  train 中保留的 gamePlayerId 筆數      : {_kept_p:,}")
    print(f"  train 中保留的 gamePlayerOtherId 筆數 : {_kept_o:,}")

    # 合併全部來源
    train_raw = pd.concat([train_main, train_proc, train_e, train_k], ignore_index=True)
    train_raw["_no_server_lbl"] = train_raw["_no_server_lbl"].fillna(False)

    train_raw = train_raw.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    test_raw  = test_raw.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    print(f"  train.csv          : {train_main.shape}")
    print(f"  processed_train_e  : {train_proc.shape}")
    print(f"  train_e.csv        : {train_e.shape}")
    print(f"  train_k.csv        : {train_k.shape}")
    print(f"  合併後訓練資料      : {train_raw.shape}")
    print(f"  test shape          : {test_raw.shape}")
    print(f"  train.csv 選手數    : {len(train_players)}")
    print(f"  test 中未見選手數   : {len((set(test_raw['gamePlayerId'].unique()) | set(test_raw['gamePlayerOtherId'].unique())) - train_players)}")

# ══════════════════════════════════════════════════════════════════
# 特徵工程
# ══════════════════════════════════════════════════════════════════
LAG_COLS = ["actionId", "pointId", "handId", "strengthId", "spinId", "positionId", "strikeId"]

def build_features(df: pd.DataFrame,
                   is_train: bool = True,
                   #match_zero_prob: float  = 0.0,
                   known_players: set = None) -> pd.DataFrame:

    df = df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    # ── 目標變數 (訓練集) ────────────────────────────────────────
    if is_train:
        df["target_actionId"]       = df.groupby("rally_uid")["actionId"].shift(-1)
        df["target_pointId"]        = df.groupby("rally_uid")["pointId"].shift(-1)
        df["target_serverGetPoint"] = df["serverGetPoint"].fillna(0).astype(int)
        df = df.dropna(subset=["target_actionId", "target_pointId"]).copy()
        df["target_actionId"] = df["target_actionId"].astype(int)
        df["target_pointId"]  = df["target_pointId"].astype(int)

    # ── 隨機將 match ID 設為 0（訓練時強化泛化） ──────────
    #if match_zero_prob > 0:
    #    df.loc[np.random.random(len(df)) < match_zero_prob, "match"] = 0

    # ── 未見選手 → 0（測試集） ───────────────────────────────────
    if not is_train and known_players is not None:
        df["gamePlayerId"]      = df["gamePlayerId"].where(df["gamePlayerId"].isin(known_players),      0)
        df["gamePlayerOtherId"] = df["gamePlayerOtherId"].where(df["gamePlayerOtherId"].isin(known_players), 0)

    # ── Lag Features（1–4 拍） ────────────────────────────────────
    for k in [1, 2, 3, 4]:
        for col in LAG_COLS:
            df[f"lag{k}_{col}"] = (df.groupby("rally_uid")[col]
                                     .shift(k).fillna(0).astype(int))

    # ── pointId → (x, y) 座標分解 ────────────────────────────────
    # 當前拍落點（= 下一擊球者的「來球方向」）
    df["cur_pt_x"] = pt_x(df["pointId"])
    df["cur_pt_y"] = pt_y(df["pointId"])
    for k in [1, 2, 3, 4]:
        df[f"lag{k}_pt_x"] = pt_x(df[f"lag{k}_pointId"])
        df[f"lag{k}_pt_y"] = pt_y(df[f"lag{k}_pointId"])

    # ── 正斜線指標（handId × 落點方向） ──────────────────────────
    # 當前拍（n-1 拍）：打球方 handId + 落在哪側 → 下一擊球者的來球方式
    df["cur_cross_court"] = cross_court_indicator(df["handId"], df["cur_pt_x"])
    for k in [1, 2, 3]:
        df[f"lag{k}_cross_court"] = cross_court_indicator(
            df[f"lag{k}_handId"], df[f"lag{k}_pt_x"])

    # ── 正反手 × 落點方向複合特徵 ────────────────────────────────
    # 直接讓模型學習不同 handId 在不同落點區域的分佈差異
    df["cur_hand_zone"]  = df["handId"]        * 10 + df["cur_pt_x"]
    df["lag1_hand_zone"] = df["lag1_handId"]   * 10 + df["lag1_pt_x"]
    df["lag2_hand_zone"] = df["lag2_handId"]   * 10 + df["lag2_pt_x"]

    # ── 來球區域轉移：上一拍 → 當前落點的 x 位移 ──────────────
    # > 0 向反手移動，< 0 向正手移動
    df["zone_shift_x"]  = df["cur_pt_x"]  - df["lag1_pt_x"]
    df["zone_shift_y"]  = df["cur_pt_y"]  - df["lag1_pt_y"]
    # 與同側選手（lag2 = 與當前拍同方向的選手）的累積位移
    df["zone_shift2_x"] = df["lag1_pt_x"] - df["lag2_pt_x"]

    # ── 局勢特徵 ─────────────────────────────────────────────────
    df["score_diff"]      = df["scoreSelf"] - df["scoreOther"]
    df["total_score"]     = df["scoreSelf"] + df["scoreOther"]
    df["is_serve"]        = (df["strikeNumber"] == 1).astype(int)
    df["is_receive"]      = (df["strikeNumber"] == 2).astype(int)
    df["is_early_rally"]  = (df["strikeNumber"] <= 3).astype(int)

    # ── Lag 可用性旗標（明確告知模型目前 lag 是真實歷史還是補 0 的假值）──
    df["lag1_available"] = (df["strikeNumber"] >= 2).astype(int)
    df["lag2_available"] = (df["strikeNumber"] >= 3).astype(int)
    df["lag3_available"] = (df["strikeNumber"] >= 4).astype(int)
    df["is_clutch"]       = (
        ((df["scoreSelf"] >= 9) | (df["scoreOther"] >= 9)) &
        (df["score_diff"].abs() <= 2)
    ).astype(int)
    df["rally_progress"]  = df["strikeNumber"] / 20.0
    df["is_server_turn"]  = (df["strikeNumber"] % 2 != 0).astype(int)

    # 推斷下一拍的 strikeId（強力約束 actionId 預測）
    # strikeNumber n → 下一拍 n+1: 1=serve只在第1拍, 2=receive在第2拍, 否則4=rally
    df["next_strikeId"] = np.where(df["strikeNumber"] == 1, 2,
                          np.where(df["strikeNumber"] == 2, 4, 4)).astype(int)

    # ── Action 大類 ───────────────────────────────────────────────
    df["action_cat"]      = df["actionId"].map(ACTION_CATEGORY).fillna(0).astype(int)
    df["lag1_action_cat"] = df["lag1_actionId"].map(ACTION_CATEGORY).fillna(0).astype(int)
    df["lag2_action_cat"] = df["lag2_actionId"].map(ACTION_CATEGORY).fillna(0).astype(int)


    # ── 複合類別特徵 ──────────────────────────────────────────────
    # 新增以下兩個特徵
    #df["hand_spin_cat"]     = df["handId"]        * 10 + df["spinId"]
    #df["lag1_act_pt_cat"]   = df["lag1_actionId"] * 10 + df["lag1_pointId"]
    #df["lag1_hand_pt_cat"]  = df["lag1_handId"]   * 10 + df["lag1_pointId"]

    # ── 對手被調動幅度（與 lag2 = 同側選手的前一拍比較） ─────────
    df["opp_disp_x"]   = (df["cur_pt_x"] - df["lag2_pt_x"]).fillna(0.0)
    df["opp_disp_y"]   = (df["cur_pt_y"] - df["lag2_pt_y"]).fillna(0.0)
    df["opp_disp_dist"]= np.sqrt(df["opp_disp_x"]**2 + df["opp_disp_y"]**2)

    # ── 壓力分 / 比分態勢 ────────────────────────────────────────
    df["is_deuce"]    = ((df["scoreSelf"] >= 10) & (df["scoreOther"] >= 10)).astype(int)
    df["is_leading"]  = (df["scoreSelf"] > df["scoreOther"]).astype(int)
    df["is_trailing"] = (df["scoreSelf"] < df["scoreOther"]).astype(int)

    # ── 落點深淺 / 左右 二值旗標 ────────────────────────────────
    # cur（cur_pt_x/y 已算好：x=1正手 2中 3反手；y=1短 2半出台 3長）
    df["cur_is_short"]     = (df["cur_pt_y"] == 1).astype(int)
    df["cur_is_half_long"] = (df["cur_pt_y"] == 2).astype(int)
    df["cur_is_long"]      = (df["cur_pt_y"] == 3).astype(int)
    df["cur_to_forehand"]  = (df["cur_pt_x"] == 1).astype(int)
    df["cur_to_middle"]    = (df["cur_pt_x"] == 2).astype(int)
    df["cur_to_backhand"]  = (df["cur_pt_x"] == 3).astype(int)
    # lag1
    df["lag1_is_short"]     = (df["lag1_pt_y"] == 1).astype(int)
    df["lag1_is_half_long"] = (df["lag1_pt_y"] == 2).astype(int)
    df["lag1_is_long"]      = (df["lag1_pt_y"] == 3).astype(int)
    df["lag1_to_forehand"]  = (df["lag1_pt_x"] == 1).astype(int)
    df["lag1_to_middle"]    = (df["lag1_pt_x"] == 2).astype(int)
    df["lag1_to_backhand"]  = (df["lag1_pt_x"] == 3).astype(int)
    # lag2
    df["lag2_is_short"]    = (df["lag2_pt_y"] == 1).astype(int)
    df["lag2_is_long"]     = (df["lag2_pt_y"] == 3).astype(int)
    df["lag2_to_forehand"] = (df["lag2_pt_x"] == 1).astype(int)
    df["lag2_to_backhand"] = (df["lag2_pt_x"] == 3).astype(int)

    # ── 落點是否改變（lag1=0 表示無前拍，視為未改變） ────────────
    df["point_changed"]  = ((df["pointId"] != df["lag1_pointId"]) & (df["lag1_pointId"] != 0)).astype(int)
    df["zone_x_changed"] = ((df["cur_pt_x"] != df["lag1_pt_x"])   & (df["lag1_pt_x"]   != 0)).astype(int)
    df["zone_y_changed"] = ((df["cur_pt_y"] != df["lag1_pt_y"])   & (df["lag1_pt_y"]   != 0)).astype(int)

    # ── Rally 內累計落點分佈（shift=1，不含當前拍） ──────────────
    for _col, _vals in [
        ("count_short",         [1, 2, 3]),
        ("count_half_long",     [4, 5, 6]),
        ("count_long",          [7, 8, 9]),
        ("count_forehand_zone", [1, 4, 7]),
        ("count_middle_zone",   [2, 5, 8]),
        ("count_backhand_zone", [3, 6, 9]),
    ]:
        _tmp = f"__tmp_{_col}"
        df[_tmp] = df["pointId"].isin(_vals).astype(int)
        df[_col] = (
            df.groupby("rally_uid")[_tmp]
              .transform(lambda x: x.shift(1).fillna(0).cumsum())
              .astype(int)
        )
    df = df.drop(columns=[c for c in df.columns if c.startswith("__tmp_")])

    # ── 轉移編碼：point_transition 3-gram（lag2→lag1→cur）; spin 2-gram ──
    df["point_transition"] = np.where(
        (df["lag2_pointId"] > 0) & (df["lag1_pointId"] > 0),
        df["lag2_pointId"] * 10000 + df["lag1_pointId"] * 100 + df["pointId"],
        np.where(df["lag1_pointId"] > 0,
                 df["lag1_pointId"] * 100 + df["pointId"], 0)
    ).astype(int)
    df["spin_transition"] = np.where(
        df["lag1_spinId"] > 0, df["lag1_spinId"] * 100 + df["spinId"], 0
    ).astype(int)

    # ── 3-gram 序列特徵（lag2→lag1→cur）────────────────────────────
    # action_transition / point_transition 皆為 3-gram；spin_3gram 亦同
    df["action_transition"] = np.where(
        (df["lag2_actionId"] > 0) & (df["lag1_actionId"] > 0),
        df["lag2_actionId"] * 10000 + df["lag1_actionId"] * 100 + df["actionId"],
        np.where(df["lag1_actionId"] > 0,
                 df["lag1_actionId"] * 100 + df["actionId"], 0)
    ).astype(int)
    df["spin_3gram"] = np.where(
        (df["lag2_spinId"] > 0) & (df["lag1_spinId"] > 0),
        df["lag2_spinId"] * 10000 + df["lag1_spinId"] * 100 + df["spinId"],
        np.where(df["lag1_spinId"] > 0,
                 df["lag1_spinId"] * 100 + df["spinId"], 0)
    ).astype(int)

    # ── 深淺位移 lag2 → lag1（補 zone_shift2_x 的 y 方向） ──────
    df["zone_shift2_y"] = df["lag1_pt_y"] - df["lag2_pt_y"]

    # ── 連續性與變化特徵（Feature 15）──────────────────────────
    # 是否與前一拍相同（last=lag1, prev=lag2），lag2=0 表示無資料視為不同
    df["same_action_as_prev"] = ((df["lag1_actionId"] == df["lag2_actionId"]) & (df["lag2_actionId"] > 0)).astype(int)
    df["same_point_as_prev"]  = ((df["lag1_pointId"]  == df["lag2_pointId"])  & (df["lag2_pointId"]  > 0)).astype(int)
    df["same_spin_as_prev"]   = ((df["lag1_spinId"]   == df["lag2_spinId"])   & (df["lag2_spinId"]   > 0)).astype(int)
    df["same_depth_as_prev"]  = ((df["lag1_pt_y"]     == df["lag2_pt_y"])     & (df["lag2_pt_y"]     > 0)).astype(int)
    df["same_zone_as_prev"]   = ((df["lag1_pt_x"]     == df["lag2_pt_x"])     & (df["lag2_pt_x"]     > 0)).astype(int)

    # 最近 5 拍的變化次數（cur + lag1~lag4，4 個相鄰差異）
    df["num_point_changes_last5"] = (
        ((df["pointId"]      != df["lag1_pointId"]) & (df["lag1_pointId"] > 0)).astype(int) +
        ((df["lag1_pointId"] != df["lag2_pointId"]) & (df["lag2_pointId"] > 0)).astype(int) +
        ((df["lag2_pointId"] != df["lag3_pointId"]) & (df["lag3_pointId"] > 0)).astype(int) +
        ((df["lag3_pointId"] != df["lag4_pointId"]) & (df["lag4_pointId"] > 0)).astype(int)
    )
    df["num_zone_changes_last5"] = (
        ((df["cur_pt_x"]  != df["lag1_pt_x"]) & (df["lag1_pt_x"] > 0)).astype(int) +
        ((df["lag1_pt_x"] != df["lag2_pt_x"]) & (df["lag2_pt_x"] > 0)).astype(int) +
        ((df["lag2_pt_x"] != df["lag3_pt_x"]) & (df["lag3_pt_x"] > 0)).astype(int) +
        ((df["lag3_pt_x"] != df["lag4_pt_x"]) & (df["lag4_pt_x"] > 0)).astype(int)
    )

    # ── serveId / serveNumber → rally 層級廣播 ───────────────────────
    # 用 groupby().first() 確保 rally_uid 唯一（原始資料 strikeNumber 可能重複）
    _s1 = df[df["strikeNumber"] == 1].groupby("rally_uid")

    if "serveNumber" in df.columns:
        _sn = _s1["serveNumber"].first()
        df["rally_serve_number"] = df["rally_uid"].map(_sn).fillna(0).astype(int)
    else:
        df["rally_serve_number"] = 0

    if "serveId" in df.columns:
        _si = _s1["serveId"].first()
        df["rally_serve_id"] = df["rally_uid"].map(_si).fillna(0).astype(int)
    else:
        df["rally_serve_id"] = 0

    # ── positionId rally 廣播（發球站位 / 接球站位）────────────────
    _p1 = df[df["strikeNumber"] == 1].groupby("rally_uid")["positionId"].first()
    _p2 = df[df["strikeNumber"] == 2].groupby("rally_uid")["positionId"].first()
    df["rally_serve_pos"]   = df["rally_uid"].map(_p1).fillna(0).astype(int)
    df["rally_receive_pos"] = df["rally_uid"].map(_p2).fillna(0).astype(int)
    # strikeNumber==1 的行是在預測第二拍，第二拍的接球站位尚未發生，避免資料洩漏
    df.loc[df["strikeNumber"] == 1, "rally_receive_pos"] = 0
    df["serve_receive_pos_combo"] = df["rally_serve_pos"] * 10 + df["rally_receive_pos"]

    return df

# ══════════════════════════════════════════════════════════════════
# 4. 特徵欄位定義
# ══════════════════════════════════════════════════════════════════
CONTEXT_FEATS = [
    "strikeNumber", "sex", "numberGame",
    # 選手 ID（train.csv 有值，其餘來源 / 歸零比賽設為 0）
    "gamePlayerId", "gamePlayerOtherId",
    # 當下比分、差距、總和
    "scoreSelf", "scoreOther", "score_diff", "total_score",
    # 當下小分是發球、接發、前3拍、多拍來回。
    "is_serve", "is_receive", "is_early_rally", "is_clutch",
    # Lag 可用性（strikeNumber >= 2/3/4 時各 lag 才有真實值）
    #"lag1_available", "lag2_available", "lag3_available",
    # 當下戰局狀況
    "is_deuce", #"is_leading", "is_trailing",
    "rally_progress", "is_server_turn", #"next_strikeId",
    # 發球局勢（第幾次發球 / 發球類型），只在 train_e/train_k 來源有值
    #"rally_serve_number", "rally_serve_id",
    # 發球與接球站位（positionId 在第 1、2 拍有值，廣播到全 rally）
    "rally_serve_pos", "rally_receive_pos", "serve_receive_pos_combo",
]

# 當前拍（n-1）特徵 = 給下一拍提供最直接的來球資訊
CURRENT_FEATS = [
    "handId", "strengthId", "spinId", "positionId", "strikeId",
    "actionId", "pointId",
    "cur_pt_x", "cur_pt_y",          # 來球落點座標（下一擊球者視角）
    "cur_cross_court", "cur_hand_zone",
    "action_cat",
    "zone_shift_x", "zone_shift_y",
    "opp_disp_x", "opp_disp_y", "opp_disp_dist",
    "cur_is_short", "cur_is_half_long", "cur_is_long",
    "cur_to_forehand", "cur_to_middle", "cur_to_backhand",
    "point_changed", "zone_x_changed", "zone_y_changed",
    "count_short", "count_half_long", "count_long",
    "count_forehand_zone", "count_middle_zone", "count_backhand_zone",
    "same_action_as_prev", "same_point_as_prev", "same_spin_as_prev",
    "same_depth_as_prev", "same_zone_as_prev",
    "num_point_changes_last5", "num_zone_changes_last5",
]

LAG1_FEATS = [
    "lag1_actionId", "lag1_pointId", "lag1_handId", "lag1_strengthId",
    "lag1_spinId", "lag1_positionId", "lag1_strikeId",
    "lag1_pt_x", "lag1_pt_y",
    "lag1_cross_court", "lag1_hand_zone",
    "lag1_action_cat",
    "lag1_is_short", "lag1_is_half_long", "lag1_is_long",
    "lag1_to_forehand", "lag1_to_middle", "lag1_to_backhand",
    "action_transition", "point_transition", "spin_transition",
]

LAG23_FEATS = [
    "lag2_actionId", "lag2_pointId", "lag2_handId", "lag2_strengthId",
    "lag2_spinId", "lag2_positionId", "lag2_strikeId",
    "lag2_pt_x", "lag2_pt_y",
    "lag2_cross_court", "lag2_hand_zone", "lag2_action_cat",
    "zone_shift2_x", "zone_shift2_y",
    "lag2_is_short", "lag2_is_long", "lag2_to_forehand", "lag2_to_backhand",
    "lag3_actionId", "lag3_pointId", "lag3_handId", "lag3_strengthId",
    "lag3_spinId", "lag3_positionId", "lag3_strikeId",
    "lag3_pt_x", "lag3_pt_y", "lag3_cross_court",
    # 3-gram 序列特徵（action_transition/point_transition 已在 LAG1_FEATS）
    "spin_3gram",
]

# 球種模型（前 2 拍：current + lag1）
ACTION_FEATS = CONTEXT_FEATS + CURRENT_FEATS + LAG1_FEATS + LAG23_FEATS

# 落點模型（前 4 拍：current + lag1 + lag2 + lag3）
POINT_FEATS  = CONTEXT_FEATS + CURRENT_FEATS + LAG1_FEATS + LAG23_FEATS

# 勝負模型（同球種模型特徵）
SERVER_FEATS = ACTION_FEATS

def filter_feats(df, feats, name):
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"  [!] {name} 缺少欄位: {missing}")
    return [f for f in feats if f in df.columns]

if __name__ == "__main__":
    # ══════════════════════════════════════════════════════════════════
    # 5. 建立特徵
    # ══════════════════════════════════════════════════════════════════
    print("建立訓練特徵...")
    train_df = build_features(
        train_raw.copy(),
        is_train=True,
        #match_zero_prob=MATCH_ZERO_PROB,
    )
    # incoming_zone 是 cur_pt_x 的別名，需要在 df 中存在
    train_df["incoming_zone"] = train_df["cur_pt_x"]
    # 標記每個 rally 的最後一拍（對齊 test 的 tail(1) 評估方式）
    _last_idx = train_df.groupby("rally_uid").tail(1).index
    train_df["_is_last_in_rally"] = train_df.index.isin(_last_idx)
    print(f"  訓練樣本: {len(train_df)}")

    print("建立測試特徵...")
    test_df = build_features(
        test_raw.copy(),
        is_train=False,
        known_players=train_players,
    )
    test_df["incoming_zone"] = test_df["cur_pt_x"]
    # 每個 rally 取最後一拍（= 最新一筆已知資訊，預測下一拍）
    test_last = test_df.groupby("rally_uid").tail(1).reset_index(drop=True)
    print(f"  測試 rallies: {len(test_last)}")

    # 確認特徵欄位存在
    ACTION_FEATS = filter_feats(train_df, ACTION_FEATS, "action")
    POINT_FEATS  = filter_feats(train_df, POINT_FEATS,  "point")
    SERVER_FEATS = filter_feats(train_df, SERVER_FEATS,  "server")

    # ══════════════════════════════════════════════════════════════════
    # 6. Sample Weights
    # ══════════════════════════════════════════════════════════════════
    ACTION_MAX_W = 12.0
    POINT_MAX_W  = 8.0

    # 針對不平衡類別的個別調整
    ACTION_OVERRIDES = {
        8:  30.0,  # 拱球（稀有）
        9:  20.0,  # 磕球（稀有）
        14: 20.0,  # 放高球（稀有）
        3:   10.0,  # 殺球
        16:  1.0,  # 極稀有，放棄
        17:  1.0,
        18:  1.0,
    }
    POINT_OVERRIDES = {
        3: 40.0,   # 反手位短球（極稀少）
        1: 20.0,   # 正手位短球（少）
        0: 0.25,   # 出界/無（過多，壓低）
    }

    def make_weights(y, strike_num, max_w, overrides=None):
        classes, counts = np.unique(y, return_counts=True)
        raw_w = len(y) / (len(classes) * counts)
        raw_w = raw_w / raw_w.min()
        raw_w = np.clip(raw_w, 1.0, max_w)
        cls2w = dict(zip(classes, raw_w))
        if overrides:
            cls2w.update(overrides)
        w = np.array([cls2w.get(c, 1.0) for c in y], dtype=np.float32)
        # 短拍加權：對齊 test 分佈（strikeNumber=1 占 27.5%，strikeNumber=2 占 25.7%）
        w[strike_num == 1] *= SHORT_RALLY_BOOST_1
        w[strike_num == 2] *= SHORT_RALLY_BOOST_2
        return w

    # ══════════════════════════════════════════════════════════════════
    # 7. LightGBM 超參數
    # ══════════════════════════════════════════════════════════════════
    _BASE = dict(
        learning_rate    = 0.05,#降低0.01
        subsample_freq   = 1,
        subsample        = 0.85,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1, #1到3防止過你和
        random_state     = 32,
        verbose          = -1,
        n_jobs           = -1,
    )
    PARAMS_ACTION = {**_BASE, "objective": "multiclass", "num_class": 19,
                     "metric": "multi_logloss", "num_leaves": 127, "n_estimators": 5000}
    PARAMS_POINT  = {**_BASE, "objective": "multiclass", "num_class": 10,
                     "metric": "multi_logloss", "num_leaves": 127,  "n_estimators": 5000}
    PARAMS_SERVER = {**_BASE, "objective": "binary",
                     "metric": "auc",           "num_leaves": 63,  "n_estimators": 6000}

    # ══════════════════════════════════════════════════════════════════
    # 8. 訓練（GroupKFold by match）
    # ══════════════════════════════════════════════════════════════════
    import joblib

    y_action    = train_df["target_actionId"].values
    y_point     = train_df["target_pointId"].values
    y_server    = train_df["target_serverGetPoint"].values
    groups      = train_df["match"].values
    # train_e 的 rows 沒有 server 標籤，server 模型排除它們
    has_srv_lbl = (~train_df["_no_server_lbl"].fillna(False).astype(bool)).values

    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=32) # 有改這個
    cb  = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)]

    oof_action_pred = np.zeros(len(train_df), dtype=int)
    oof_point_pred  = np.zeros(len(train_df), dtype=int)
    oof_server_prob = np.zeros(len(train_df))

    models_action, models_point, models_server = [], [], []
    fold_scores = []

    print(f"\n{'='*60}")
    print(f"  GroupKFold 訓練（n_splits={N_FOLDS}, groups=match）")
    print(f"{'='*60}")

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_action, groups)):

        X_tr_a = train_df.iloc[tr_idx][ACTION_FEATS]
        X_va_a = train_df.iloc[va_idx][ACTION_FEATS]
        X_tr_p = train_df.iloc[tr_idx][POINT_FEATS]
        X_va_p = train_df.iloc[va_idx][POINT_FEATS]

        y_tr_a, y_va_a = y_action[tr_idx], y_action[va_idx]
        y_tr_p, y_va_p = y_point[tr_idx],  y_point[va_idx]
        y_tr_s, y_va_s = y_server[tr_idx], y_server[va_idx]

        sn_tr = train_df.iloc[tr_idx]["strikeNumber"].values
        w_a = make_weights(y_tr_a, sn_tr, ACTION_MAX_W, ACTION_OVERRIDES)
        w_p = make_weights(y_tr_p, sn_tr, POINT_MAX_W,  POINT_OVERRIDES)

        # Model A: actionId
        dtr_a = lgb.Dataset(X_tr_a, label=y_tr_a, weight=w_a, free_raw_data=False)
        dva_a = dtr_a.create_valid(X_va_a, y_va_a)
        ma    = lgb.train(PARAMS_ACTION, dtr_a, valid_sets=[dva_a], callbacks=cb)

        # Model P: pointId
        dtr_p = lgb.Dataset(X_tr_p, label=y_tr_p, weight=w_p, free_raw_data=False)
        dva_p = dtr_p.create_valid(X_va_p, y_va_p)
        mp    = lgb.train(PARAMS_POINT, dtr_p, valid_sets=[dva_p], callbacks=cb)

        # Model S: serverGetPoint（排除 train_e 的無標籤 rows）
        srv_tr = has_srv_lbl[tr_idx]
        srv_va = has_srv_lbl[va_idx]
        dtr_s = lgb.Dataset(X_tr_a.iloc[srv_tr], label=y_tr_s[srv_tr], free_raw_data=False)
        dva_s = dtr_s.create_valid(X_va_a.iloc[srv_va], y_va_s[srv_va])
        ms    = lgb.train(PARAMS_SERVER, dtr_s, valid_sets=[dva_s], callbacks=cb)

        # OOF 預測
        raw_a = ma.predict(X_va_a)             # (n, 19)
        raw_p = mp.predict(X_va_p)             # (n, 10)
        # server 只預測有標籤的 rows
        p_srv_full = np.zeros(len(va_idx))
        p_srv_full[srv_va] = ms.predict(X_va_a.iloc[srv_va])
        p_srv = p_srv_full

        p_act = raw_a.argmax(axis=1).astype(int)
        p_act[np.isin(p_act, [15, 16, 17, 18])] = 0   # 發球類別：test至少給一拍，不可能預測發球
        p_pt  = raw_p.argmax(axis=1).astype(int)

        oof_action_pred[va_idx] = p_act
        oof_point_pred[va_idx]  = p_pt
        oof_server_prob[va_idx] = p_srv

        f1_a = f1_score(y_va_a, p_act, average="macro", zero_division=0)
        f1_p = f1_score(y_va_p, p_pt,  average="macro", zero_division=0)
        auc  = roc_auc_score(y_va_s[srv_va], p_srv[srv_va])
        sc   = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
        fold_scores.append(sc)

        print(f"  Fold {fold+1}  Score={sc:.4f}  Action={f1_a:.4f}  Point={f1_p:.4f}  AUC={auc:.4f}")
        models_action.append(ma)
        models_point.append(mp)
        models_server.append(ms)

    # ── OOF 全局評估 ──────────────────────────────────────────────────
    oof_f1_a = f1_score(y_action, oof_action_pred, average="macro", zero_division=0)
    oof_f1_p = f1_score(y_point,  oof_point_pred,  average="macro", zero_division=0)
    oof_auc  = roc_auc_score(y_server[has_srv_lbl], oof_server_prob[has_srv_lbl])
    oof_sc   = 0.4 * oof_f1_a + 0.4 * oof_f1_p + 0.2 * oof_auc

    print(f"\n{'='*60}")
    print(f"  CV Mean  : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  OOF Score: {oof_sc:.4f}")
    print(f"  ├─ Action Macro-F1 : {oof_f1_a:.4f}")
    print(f"  ├─ Point  Macro-F1 : {oof_f1_p:.4f}")
    print(f"  └─ Winner AUC      : {oof_auc:.4f}")
    print(f"{'='*60}")

    # ── Per-strikeNumber 答對率分析 ───────────────────────────────────
    sn_all = train_df["strikeNumber"].values
    print(f"\n  Per-strikeNumber 答對率（預測第 strikeNumber+1 拍）:")
    print(f"  {'sn':>4} {'樣本數':>8} {'action_acc':>11} {'point_acc':>10}")
    for sn in sorted(np.unique(sn_all)):
        mask = sn_all == sn
        n = mask.sum()
        acc_a = (oof_action_pred[mask] == y_action[mask]).mean()
        acc_p = (oof_point_pred[mask]  == y_point[mask]).mean()
        print(f"  {sn:4d} {n:8,} {acc_a:11.4f} {acc_p:10.4f}")

    # ── Per-class 分析（調參用）────────────────────────────────────────
    from sklearn.metrics import precision_score, recall_score

    act_f1   = f1_score(y_action, oof_action_pred, average=None, zero_division=0)
    act_prec = precision_score(y_action, oof_action_pred, average=None, zero_division=0)
    act_rec  = recall_score(y_action, oof_action_pred, average=None, zero_division=0)
    pt_f1    = f1_score(y_point, oof_point_pred, average=None, zero_division=0)
    pt_prec  = precision_score(y_point, oof_point_pred, average=None, zero_division=0)
    pt_rec   = recall_score(y_point, oof_point_pred, average=None, zero_division=0)

    def print_per_class(f1_arr, prec_arr, rec_arr, counts_arr, label):
        print(f"\n  Per-class — {label}:")
        print(f"  {'cls':>4} {'count':>7} {'prec':>7} {'rec':>7} {'F1':>7}")
        for c, (p, r, f, cnt) in enumerate(zip(prec_arr, rec_arr, f1_arr, counts_arr)):
            flag = " ← low-rec" if r < 0.10 and cnt > 0 else (
                   " ← low-prec" if p < 0.10 and cnt > 0 else "")
            print(f"  {c:4d} {cnt:7,} {p:7.4f} {r:7.4f} {f:7.4f}{flag}")

    act_counts = np.bincount(y_action, minlength=len(act_f1))
    pt_counts  = np.bincount(y_point,  minlength=len(pt_f1))
    print_per_class(act_f1, act_prec, act_rec, act_counts, "actionId (19 classes)")
    print_per_class(pt_f1,  pt_prec,  pt_rec,  pt_counts,  "pointId  (10 classes)")

    # ── Feature Importance（5 fold 平均，gain 基準）──────────────────────
    def print_importance(models, feat_cols, label, top_n=30):
        imp = np.zeros(len(feat_cols))
        for m in models:
            imp += m.feature_importance(importance_type="gain")
        imp /= len(models)
        order = np.argsort(imp)[::-1]
        print(f"\n  Feature Importance — {label} (top {top_n}, gain avg):")
        print(f"  {'rank':>5} {'feature':<35} {'gain':>10}")
        for rank, i in enumerate(order[:top_n], 1):
            print(f"  {rank:5d} {feat_cols[i]:<35} {imp[i]:10.1f}")
        # 低重要性特徵（gain < 1% of max）
        threshold = imp.max() * 0.01
        low = [(feat_cols[i], imp[i]) for i in order if imp[i] < threshold]
        if low:
            print(f"\n  [!] gain < 1% of max（可考慮移除）:")
            for fname, fval in low:
                print(f"      {fname:<35} {fval:10.1f}")

    print_importance(models_action, ACTION_FEATS, "action")
    print_importance(models_point,  POINT_FEATS,  "point")
    print_importance(models_server, SERVER_FEATS,  "server")

    # ══════════════════════════════════════════════════════════════════
    # 9. 儲存模型權重
    # ══════════════════════════════════════════════════════════════════
    joblib.dump(models_action, os.path.join(MODEL_DIR, "lgbm_action_folds_unmasked.pkl"))
    joblib.dump(models_point,  os.path.join(MODEL_DIR, "lgbm_point_folds_unmasked.pkl"))
    joblib.dump(models_server, os.path.join(MODEL_DIR, "lgbm_server_folds_unmasked.pkl"))

    meta = {
        "action_feat_cols"  : ACTION_FEATS,
        "point_feat_cols"   : POINT_FEATS,
        "server_feat_cols"  : SERVER_FEATS,
        "train_players"     : train_players,
        "oof_score"         : oof_sc,
        "oof_f1_action"     : oof_f1_a,
        "oof_f1_point"      : oof_f1_p,
        "oof_auc_server"    : oof_auc,
        "fold_scores"       : fold_scores,
        "proc_rally_offset" : PROC_RALLY_OFFSET,
        "proc_match_offset" : PROC_MATCH_OFFSET,
        "proc_player_offset": PROC_PLAYER_OFFSET,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta_unmasked.pkl"))

    print(f"\n模型權重已儲存至 {MODEL_DIR}/")
    print(f"  lgbm_action_folds_unmasked.pkl  ({N_FOLDS} folds, ensemble 用)")
    print(f"  lgbm_point_folds_unmasked.pkl   ({N_FOLDS} folds, ensemble 用)")
    print(f"  lgbm_server_folds_unmasked.pkl  ({N_FOLDS} folds, ensemble 用)")
    print(f"  meta_unmasked.pkl               (特徵欄位 / OOF 分數 / 偏移量)")

    # ══════════════════════════════════════════════════════════════════
    # 10. 推論（5-fold ensemble）
    # ══════════════════════════════════════════════════════════════════
    print("\n推論測試集...")
    X_test_a = test_last[ACTION_FEATS]
    X_test_p = test_last[POINT_FEATS]

    prob_act = np.mean([m.predict(X_test_a) for m in models_action], axis=0)   # (n, 19)
    prob_pt  = np.mean([m.predict(X_test_p) for m in models_point],  axis=0)   # (n, 10)
    prob_srv = np.mean([m.predict(X_test_a) for m in models_server], axis=0)   # (n,)

    # 極稀有類別（training 樣本 < 10）強制不選
    prob_act[:, [15, 16, 17, 18]] = 0.0

    pred_act = prob_act.argmax(axis=1).astype(int)
    pred_pt  = prob_pt.argmax(axis=1).astype(int)

    submission = pd.DataFrame({
        "rally_uid":      test_last["rally_uid"].values,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": prob_srv,   # 輸出 0–1 機率，不做二值化
    })

    npz_path = os.path.join(MODEL_DIR, "proba_unmasked.npz")
    np.savez(
        npz_path,
        rally_uid = test_last["rally_uid"].values,
        action    = prob_act,    # shape (1845, 19)  各球種的機率
        point     = prob_pt,     # shape (1845, 10)  各落點的機率
        winner    = prob_srv,    # shape (1845,)      發球者得分機率
    )
    print(f"機率陣列儲存至 {npz_path}")
    out_path = os.path.join(SUBMISSION_DIR, "submission_unmasked.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission 儲存至 {out_path}  (shape: {submission.shape})")
    act_dist = pd.Series(pred_act).value_counts().sort_index()
    pt_dist  = pd.Series(pred_pt).value_counts().sort_index()

    print(f"\n  actionId 分佈 (全類別):")
    for cls, cnt in act_dist.items():
        print(f"    {cls:>3}: {cnt:>5}")
    print(f"\n  pointId 分佈 (全類別):")
    for cls, cnt in pt_dist.items():
        print(f"    {cls:>3}: {cnt:>5}")
    print(f"\n  serverGetPoint（機率）統計:")
    print(f"    min={prob_srv.min():.4f}  max={prob_srv.max():.4f}"
          f"  mean={prob_srv.mean():.4f}  median={np.median(prob_srv):.4f}")
    print(f"    >0.5 的比例: {(prob_srv > 0.5).mean():.3f}")
