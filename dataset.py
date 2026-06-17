"""
dataset.py - 資料集定義
包含：RallyDataset, collate_fn

v20: 最後一拍 pointId=0 (rally-end marker) 是 LB macro-F1 的合法 class — 原本 v19
     用 point_mask 把這些樣本從 loss 排除，導致模型永遠不預測 class 0、val PtF1 也
     看不到 class 0 → CV-LB gap 0.08。改為：
       (1) point_mask 永遠 True（LB 跟 val 用同一指標）
       (2) sliding window 階段以 cfg.last_stroke_target_prob 機率採樣 last-stroke
           target，避免每個 rally 都貢獻 1 個 class-0 樣本造成 over-prediction。
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import ACTION_TO_CATEGORY


class RallyDataset(Dataset):
    """
    桌球回合序列資料集。

    訓練模式（use_sliding_window=True）：
        每個 rally 生成多個樣本：用前 k 拍預測第 k+1 拍 (k=1,...,n-1)。
        v20: target_idx = n-1 (last stroke, pointId=0) 以 cfg.last_stroke_target_prob
        機率納入；其他 target 一律納入。所有納入的樣本 point_mask=True。

    訓練模式（use_sliding_window=False）：
        每個 rally 只生成一個樣本：用前 n-1 拍預測第 n 拍 (target = pointId=0)。
        v20: point_mask=True (供 loss 計算)。此模式只用於 val_nosw 場景下的 action/
        winner 指標；val_nosw 的 pointId 指標目標全為 0，本來就不會被當參考。

    測試模式：
        用所有給定拍次作為輸入，預測下一拍。
    """

    def __init__(self, rally_dict, cfg, mode="train", use_sliding_window=True,
                 fingerprint_dict=None):
        """
        Args:
            fingerprint_dict: Optional dict[rally_uid] -> ndarray (rally_seq_len, 2*fp_dim)
                              來自 utils.build_rally_fingerprints。提供時每個樣本會多
                              帶一個 (max_seq_len, 2*fp_dim) 的 fingerprint tensor。
        """
        self.max_seq_len = cfg.max_seq_len
        self.mode = mode
        self.feature_names = cfg.all_features
        self.n_features = cfg.n_features
        self.fingerprint_dict = fingerprint_dict
        self.fp_dim = int(getattr(cfg, "fingerprint_dim", 0))
        self.samples = []

        for uid, df in rally_dict.items():
            df = df.sort_values("strikeNumber").reset_index(drop=True)
            seq_len = len(df)
            fp_full = (
                fingerprint_dict[uid] if fingerprint_dict is not None else None
            )  # (seq_len, 2*fp_dim) or None

            if mode == "train":
                # v31.1: serverGetPoint = -1 是 sentinel（test_k 沒有真實 winner 標籤），
                # winner_mask=False 跳過 winner loss。其他正常 row 的 winner_mask=True。
                _server_raw = int(df.iloc[0]["serverGetPoint"])
                winner_known = _server_raw in (0, 1)
                server_get_point = _server_raw if winner_known else 0  # placeholder
                last_idx = seq_len - 1

                if use_sliding_window and seq_len >= 3:
                    # v20: 以機率採樣 last-stroke target (target_idx = n-1, pointId=0)。
                    # 全 train 每個 rally 都貢獻 1 個 class-0 樣本會讓模型過度傾向預測 0；
                    # 用 prob 採樣讓 class-0 比例可調。np.random 由 cfg.seed 決定，可重現。
                    last_stroke_p = float(
                        getattr(cfg, "last_stroke_target_prob", 1.0)
                    )
                    include_last = (
                        last_stroke_p >= 1.0 or np.random.rand() < last_stroke_p
                    )
                    end_idx = seq_len if include_last else (seq_len - 1)

                    for target_idx in range(1, end_idx):
                        input_df = df.iloc[:target_idx]
                        target_action = int(df.iloc[target_idx]["actionId"])
                        target_point = int(df.iloc[target_idx]["pointId"])

                        self.samples.append({
                            "rally_uid": uid,
                            "input_data": self._extract_features(input_df, self.feature_names),
                            "fingerprint": (
                                fp_full[:target_idx].astype(np.float32)
                                if fp_full is not None else None
                            ),
                            "seq_len": len(input_df),
                            "target_action": target_action,
                            "target_action_category": int(
                                ACTION_TO_CATEGORY[
                                    np.clip(target_action, 0, len(ACTION_TO_CATEGORY) - 1)
                                ]
                            ),
                            "target_point": target_point,
                            "target_winner": server_get_point,
                            # v20: 永遠 True — class 0 (rally-end) 是 LB 評分的合法 class
                            "point_mask": True,
                            # v31.1: test_k row winner 未知 → False, 跳過 winner loss
                            "winner_mask": winner_known,
                        })
                else:
                    if seq_len >= 2:
                        input_df = df.iloc[:-1]
                        target_action = int(df.iloc[-1]["actionId"])
                        target_point = int(df.iloc[-1]["pointId"])
                        fp_slice = fp_full[:-1] if fp_full is not None else None
                    else:
                        input_df = df
                        target_action = int(df.iloc[0]["actionId"])
                        target_point = int(df.iloc[0]["pointId"])
                        fp_slice = fp_full if fp_full is not None else None

                    self.samples.append({
                        "rally_uid": uid,
                        "input_data": self._extract_features(input_df, self.feature_names),
                        "fingerprint": (
                            fp_slice.astype(np.float32) if fp_slice is not None else None
                        ),
                        "seq_len": len(input_df),
                        "target_action": target_action,
                        "target_action_category": int(
                            ACTION_TO_CATEGORY[
                                np.clip(target_action, 0, len(ACTION_TO_CATEGORY) - 1)
                            ]
                        ),
                        "target_point": target_point,
                        "target_winner": server_get_point,
                        # v20: 永遠 True (loss 計算用)；va_nosw 不用 pointId 指標所以無妨
                        "point_mask": True,
                        # v31.1: test_k row winner 未知 → False
                        "winner_mask": winner_known,
                    })

            elif mode == "test":
                self.samples.append({
                    "rally_uid": uid,
                    "input_data": self._extract_features(df, self.feature_names),
                    "fingerprint": (
                        fp_full.astype(np.float32) if fp_full is not None else None
                    ),
                    "seq_len": seq_len,
                })

    def get_sample_weights(
        self, test_seq_len_counts,
        class_balance_power: float = 0.0,
        class_balance_target: str = "action",
    ):
        """
        計算每個訓練樣本的 IPW 採樣權重。

        v37 任務 1：在原 length-IPW 之上疊加 class-aware 因子，讓稀有
                  target_action 樣本被多抽到（LB Macro F1 對 19 類少數類敏感）。
                  final_weight = length_weight × class_weight，其中
                  class_weight = (1 / (freq[c] + 1))**power
                  power=0 退回舊版 v20 行為（純 length-IPW）。

        Args:
            test_seq_len_counts:  dict/Counter，test 的 rally 長度分布。
            class_balance_power:  class-aware 強度。0=關，0.5=1/sqrt(freq)（推薦），
                                  1.0=完全反比。
            class_balance_target: "action" / "point"（point 暫時等同 action）。

        Returns:
            list[float]: 與 self.samples 等長的權重列表。
        """
        from collections import Counter
        train_counts = Counter(s["seq_len"] for s in self.samples)
        total_train = sum(train_counts.values())
        total_test = sum(test_seq_len_counts.values())

        # ---- (a) length-IPW（v20 原本邏輯） ----
        length_w = []
        for s in self.samples:
            k = s["seq_len"]
            p_test  = test_seq_len_counts.get(k, 0) / total_test
            p_train = train_counts[k] / total_train
            if p_train > 0 and p_test > 0:
                length_w.append(p_test / p_train)
            elif p_test == 0:
                length_w.append(1e-6)   # 測試集沒有此長度，幾乎不要抽到
            else:
                length_w.append(1.0)    # 訓練集沒有但測試集有，給均勻權重

        # ---- (b) class-aware（v37 任務 1 + v39 任務 A3） ----
        if class_balance_power <= 0.0:
            return length_w   # power=0 → 完全等同舊版，hash/sum 對得起來

        def _cw_map(key):
            """build {class → weight} for given target key."""
            freq = Counter(s[key] for s in self.samples)
            return {
                c: (1.0 / (n + 1)) ** class_balance_power for c, n in freq.items()
            }

        # v39 A3："both" → 合併 action + point 的 class weight。
        # combined = sqrt(cw_action × cw_point)，幾何平均避免兩個任務的稀有類
        # 權重直接相乘變過度膨脹。單任務模式維持 v37 行為。
        if class_balance_target == "both":
            cw_a = _cw_map("target_action")
            cw_p = _cw_map("target_point")
            cw_values = [
                (cw_a[s["target_action"]] * cw_p[s["target_point"]]) ** 0.5
                for s in self.samples
            ]
        else:
            target_key = (
                "target_point" if class_balance_target == "point" else "target_action"
            )
            class_w_map = _cw_map(target_key)
            cw_values = [class_w_map[s[target_key]] for s in self.samples]

        # 正規化讓平均 class_weight = 1.0，避免整體 weight scale 漂移
        mean_cw = float(np.mean(cw_values)) if cw_values else 1.0
        cw_values = [w / max(mean_cw, 1e-9) for w in cw_values]

        return [lw * cw for lw, cw in zip(length_w, cw_values)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_len = sample["seq_len"]
        data = sample["input_data"]

        padded = np.zeros((self.max_seq_len, self.n_features), dtype=np.float32)
        actual_len = min(seq_len, self.max_seq_len)
        padded[:actual_len] = data[:actual_len]

        result = {
            "features": torch.tensor(padded),
            "seq_len": actual_len,
            "rally_uid": sample["rally_uid"],
        }

        if self.fingerprint_dict is not None and sample.get("fingerprint") is not None:
            fp = sample["fingerprint"]
            fp_padded = np.zeros((self.max_seq_len, 2 * self.fp_dim), dtype=np.float32)
            fp_padded[:actual_len] = fp[:actual_len]
            result["fingerprint"] = torch.tensor(fp_padded)

        if self.mode == "train":
            result["target_action"] = sample["target_action"]
            result["target_action_category"] = sample["target_action_category"]
            result["target_point"] = sample["target_point"]
            result["target_winner"] = sample["target_winner"]
            result["point_mask"] = sample["point_mask"]
            result["winner_mask"] = sample["winner_mask"]

        return result

    @staticmethod
    def _extract_features(input_df, feature_names):
        """
        v21: 依 input prefix 即時計算 rallyProgress 後抽取特徵陣列。
        用 prefix 的 max(strikeNumber) 當分母 → train/test 每筆的 last input
        rallyProgress 都 = 1.0，消除 v18~v20 的 train/test distribution shift bug
        （v20 模型把「rallyProgress 高 = rally 結束」捷徑套到 test 1.0 → 99.7% 全
        預測 pointId=0 → LB 0.26 大跌）。utils.add_engineered_features 把 rallyProgress
        設成 0.0 placeholder，最終值在這裡覆寫。
        """
        input_df = input_df.copy()
        prefix_max = max(int(input_df["strikeNumber"].max()), 1)
        input_df["rallyProgress"] = input_df["strikeNumber"] / prefix_max
        return input_df[feature_names].values.astype(np.float32)


def collate_fn(batch):
    """自定義 batch 組裝函數"""
    features = torch.stack([b["features"] for b in batch])
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    rally_uids = [b["rally_uid"] for b in batch]

    result = {
        "features": features,
        "seq_lens": seq_lens,
        "rally_uids": rally_uids,
    }

    if "fingerprint" in batch[0]:
        result["fingerprint"] = torch.stack([b["fingerprint"] for b in batch])

    if "target_action" in batch[0]:
        result["target_action"] = torch.tensor(
            [b["target_action"] for b in batch], dtype=torch.long
        )
        result["target_action_category"] = torch.tensor(
            [b["target_action_category"] for b in batch], dtype=torch.long
        )
        result["target_point"] = torch.tensor(
            [b["target_point"] for b in batch], dtype=torch.long
        )
        result["target_winner"] = torch.tensor(
            [b["target_winner"] for b in batch], dtype=torch.float32
        )
        result["point_mask"] = torch.tensor(
            [b["point_mask"] for b in batch], dtype=torch.bool
        )
        # v31.1: winner_mask (False for test_k rows, True for others)
        result["winner_mask"] = torch.tensor(
            [b.get("winner_mask", True) for b in batch], dtype=torch.bool
        )

    return result
