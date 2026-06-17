"""
model.py - 模型架構定義
包含：PositionalEncoding, FeatureEmbedding, MultiTaskTransformer
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def _force_math_sdpa():
    """
    強制 PyTorch SDPA 使用 math 後端，避免 Flash / Memory-Efficient kernel
    在遇到 3D float attention mask（ALiBi + padding）時產生全 NaN。

    為什麼 train 正常 eval NaN：
      - train: dropout>0 → SDPA 自動 fallback 到 math 後端，mask 正確處理
      - eval:  dropout=0 → SDPA 升級到 Flash/MemEfficient，這兩個 fused
                kernel 不支援任意 3D float mask → 輸出全 NaN

    同時支援新舊版 PyTorch API。
    """
    # 新 API (PyTorch 2.1+)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.MATH]):
            yield
        return
    except ImportError:
        pass
    # 舊 API (PyTorch 2.0)
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield
        return
    except Exception:
        pass
    # 最舊版本：直接 yield，無法強制
    yield


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FeatureEmbedding(nn.Module):
    """
    將原始特徵轉換為 d_model 維度的向量表示。
    - 類別特徵 → Embedding lookup
    - 連續特徵 → BatchNorm + Linear projection
    - 最後拼接並投射到 d_model
    """

    # 哪些類別特徵屬於「player identity」、適用 rally-level 隨機 mask
    PLAYER_ID_FEATURES = ("gamePlayerId", "gamePlayerOtherId")

    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.embed_dim
        self.feature_names = cfg.all_features

        # 類別特徵 Embedding (包含 raw_features 和 engineered_features 中的類別特徵)
        self.embeddings = nn.ModuleDict()
        cat_total_dim = 0
        for feat_name, n_classes in cfg.categorical_features.items():
            self.embeddings[feat_name] = nn.Embedding(
                n_classes + 2, embed_dim, padding_idx=0
            )
            cat_total_dim += embed_dim

        # 連續特徵：排除那些已被列為類別特徵的 engineered features
        self._pure_continuous = [
            f for f in cfg.continuous_features + cfg.engineered_features
            if f not in cfg.categorical_features
        ]
        n_continuous = len(self._pure_continuous)
        # 原本用 BatchNorm1d，但 WeightedRandomSampler 過度抽短序列時 padding 0
        # 會污染 BN 的 running stats，eval 模式下除以被污染的 variance 造成 NaN。
        # LayerNorm 沒有 running stats，train/eval 行為一致，天然免疫這個問題。
        self.continuous_ln = nn.LayerNorm(n_continuous)
        self.continuous_proj = nn.Linear(n_continuous, embed_dim)
        cont_dim = embed_dim

        # 拼接後投射到 d_model
        total_dim = cat_total_dim + cont_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 預計算索引
        self._cont_indices = [
            self.feature_names.index(f) for f in self._pure_continuous
        ]
        self._cat_info = {
            feat_name: (self.feature_names.index(feat_name), n_classes)
            for feat_name, n_classes in cfg.categorical_features.items()
        }

        # Phase A: rally-level random masking of player ID（generalization 機制）
        self.player_id_dropout = float(getattr(cfg, "player_id_dropout", 0.0))
        self._player_id_set = set(self.PLAYER_ID_FEATURES) & set(self._cat_info.keys())

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        embedded_parts = []

        # Rally-level player ID dropout（player ID 在一個 rally 內僅二值，
        # 整 rally 一起 mask 比單拍 mask 更貼近真實「不知道是哪位選手」的場景）
        if (self.training and self.player_id_dropout > 0.0
                and len(self._player_id_set) > 0):
            drop = torch.rand(batch_size, device=x.device) < self.player_id_dropout
        else:
            drop = None

        # 類別特徵
        for feat_name, (idx, n_cls) in self._cat_info.items():
            vals = x[:, :, idx].long().clamp(0, n_cls + 1)
            if drop is not None and feat_name in self._player_id_set:
                vals = vals.masked_fill(drop.view(batch_size, 1), 0)
            embedded_parts.append(self.embeddings[feat_name](vals))

        # 連續特徵
        cont_feats = x[:, :, self._cont_indices]
        cont_normed = self.continuous_ln(cont_feats)  # (B, L, n_cont)
        embedded_parts.append(self.continuous_proj(cont_normed))

        combined = torch.cat(embedded_parts, dim=-1)
        return self.projection(combined)


class WinnerShortPrefixExpert(nn.Module):
    """
    Winner-only short-prefix expert.

    test_new contains many 1~2 stroke prefixes. In that regime a sequence model
    has little temporal signal, so this module lets a small MLP specialize on
    short prefixes and only correct the base winner logit when seq_len <= k.
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        dropout = cfg.dropout
        self.short_len = int(getattr(cfg, "winner_short_prefix_len", 2))
        self.bucket_dim = 4  # 1, 2, 3-4, 5+ strokes

        in_dim = d_model + self.bucket_dim
        self.expert = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, max(d_model // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 16), 1),
        )

    def _bucket_onehot(self, seq_lens):
        bucket = torch.where(
            seq_lens <= 1, 0,
            torch.where(seq_lens == 2, 1, torch.where(seq_lens <= 4, 2, 3)),
        )
        return F.one_hot(bucket.clamp(0, self.bucket_dim - 1),
                         num_classes=self.bucket_dim).float()

    def forward(self, winner_repr, seq_lens, base_logit):
        b = self._bucket_onehot(seq_lens)
        h = torch.cat([winner_repr, b], dim=-1)
        expert_logit = self.expert(h).squeeze(-1)
        gate = torch.sigmoid(self.gate(h)).squeeze(-1)
        gate = gate * (seq_lens <= self.short_len).float()
        return base_logit + gate * (expert_logit - base_logit)


class ActionShortPrefixExpert(nn.Module):
    """
    v38 任務 B：action 版的短前綴專家，鏡像 WinnerShortPrefixExpert。

    test_new 有 53% 是 1-2 拍 prefix（27.5% 第 1 拍 + 25.7% 第 2 拍），這段
    Transformer 的 sequence signal 弱。讓一個小 MLP 對短 prefix 上 action
    base_logits 做 per-class 修正；長 prefix 時 gate ≈ 0 自然 fallback。

    與 winner 版差異：輸出是 (B, n_action) 多類 logits（不是單一純量），
    gate 仍是單一 sigmoid 純量，broadcast 到所有 action class。
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        dropout = cfg.dropout
        n_action = cfg.n_action_classes
        self.short_len = int(getattr(cfg, "action_short_prefix_len", 2))
        self.bucket_dim = 4  # 1, 2, 3-4, 5+ strokes

        in_dim = d_model + self.bucket_dim
        self.expert = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_action),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, max(d_model // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 16), 1),
        )

    def _bucket_onehot(self, seq_lens):
        bucket = torch.where(
            seq_lens <= 1, 0,
            torch.where(seq_lens == 2, 1, torch.where(seq_lens <= 4, 2, 3)),
        )
        return F.one_hot(bucket.clamp(0, self.bucket_dim - 1),
                         num_classes=self.bucket_dim).float()

    def forward(self, action_repr, seq_lens, base_logits):
        b = self._bucket_onehot(seq_lens)
        h = torch.cat([action_repr, b], dim=-1)
        expert_logits = self.expert(h)                                # (B, n_action)
        gate = torch.sigmoid(self.gate(h))                            # (B, 1)
        gate = gate * (seq_lens <= self.short_len).float().unsqueeze(-1)
        return base_logits + gate * (expert_logits - base_logits)


class PointShortPrefixExpert(nn.Module):
    """
    v39 任務 A2：pointId 版的短前綴專家，鏡像 ActionShortPrefixExpert。

    test_new 53% 是 1-2 拍 prefix，pointId 在這段 Transformer 的 sequence signal
    弱。讓一個小 MLP 對短 prefix 上 point base_logits 做 per-class 修正；長 prefix
    時 gate ≈ 0 自然 fallback。輸出 (B, n_point) 多類 logits（n_point=10）。
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        dropout = cfg.dropout
        n_point = cfg.n_point_classes
        self.short_len = int(getattr(cfg, "point_short_prefix_len", 2))
        self.bucket_dim = 4  # 1, 2, 3-4, 5+ strokes

        in_dim = d_model + self.bucket_dim
        self.expert = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_point),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, max(d_model // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 16), 1),
        )

    def _bucket_onehot(self, seq_lens):
        bucket = torch.where(
            seq_lens <= 1, 0,
            torch.where(seq_lens == 2, 1, torch.where(seq_lens <= 4, 2, 3)),
        )
        return F.one_hot(bucket.clamp(0, self.bucket_dim - 1),
                         num_classes=self.bucket_dim).float()

    def forward(self, point_repr, seq_lens, base_logits):
        b = self._bucket_onehot(seq_lens)
        h = torch.cat([point_repr, b], dim=-1)
        expert_logits = self.expert(h)                                # (B, n_point)
        gate = torch.sigmoid(self.gate(h))                            # (B, 1)
        gate = gate * (seq_lens <= self.short_len).float().unsqueeze(-1)
        return base_logits + gate * (expert_logits - base_logits)


class FingerprintActionPrior(nn.Module):
    """
    v41：fingerprint → action 直接路徑（conditional-shift domain adaptation）。

    診斷：actionId 的 CV-LB gap 0.25 來自 conditional shift —— 同樣盤面下，test
    選手的動作選擇習慣跟 train 選手不同。唯一能在 test 取得的選手習慣信號 =
    fingerprint（從 test 同 match 其他 rally 算的 leave-one-out 統計，前 19 維就是
    該選手的 action 直方圖）。但 ShuttleNet 把 fingerprint 投影後「加到 stroke
    embedding」再穿過整個 encoder/decoder，這個強 prior 被深層網路稀釋；LGBM 卻
    是直接拿 action 直方圖當特徵 → 捷徑短，這正是 LGBM action 贏 NN 的關鍵。

    本模組把「下一拍 striker（= 最後 input stroke 的對手，因 gamePlayerId 每拍交替）
    的 fingerprint action 直方圖」直接投影成 action logit bias，gated residual 加到
    action head 輸出，繞過稀釋。gate 由該 striker 的 fingerprint 信心度（log_n_other）
    決定：unseen / 樣本少的選手 → gate 小 → 不亂注入噪訊（退化回原行為）。
    """

    def __init__(self, n_action, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(n_action),
            nn.Linear(n_action, n_action * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_action * 2, n_action),
        )
        # gate 由 fingerprint 信心度 (log_n_other) 決定，控制注入強度
        self.gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, next_hist, confidence):
        """
        next_hist:  (B, n_action) 下一拍 striker 的 fingerprint action 直方圖
        confidence: (B, 1) 該 striker 的 log_n_other（fingerprint 信心度）
        return:     (B, n_action) gated action logit bias（additive）
        """
        bias = self.proj(next_hist)                          # (B, n_action)
        gate = torch.sigmoid(self.gate(confidence) - 1.0)    # (B, 1)，-1 → 初始 gate≈0.27
        return gate * bias


class ManualMultiheadAttention(nn.Module):
    """
    手寫 Multi-Head Self-Attention。
    完全繞開 nn.MultiheadAttention / F.scaled_dot_product_attention，
    因此沒有 Flash / Memory-Efficient / fused fast-path 的 backend dispatch，
    train / eval 行為完全一致，3D float attention mask (ALiBi + padding)
    可以安全使用。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        """
        x: (B, L, D)
        attn_mask: (B*H, L, L) float，加到 attention scores 上
        """
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        # Attention scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        if attn_mask is not None:
            if attn_mask.dim() == 3:  # (B*H, L, L) → (B, H, L, L)
                attn_mask = attn_mask.view(B, H, L, L)
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class ManualEncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder Layer (手寫)"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = ManualMultiheadAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # Pre-LN
        h = self.norm1(x)
        h = self.self_attn(h, attn_mask)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.linear2(self.ff_dropout(F.gelu(self.linear1(h))))
        x = x + self.dropout2(h)
        return x


class ManualEncoder(nn.Module):
    """Stack of ManualEncoderLayer"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int,
                 dropout: float, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ManualEncoderLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class MultiTaskTransformer(nn.Module):
    """
    Transformer Encoder + 三個分類頭 (Multi-Task)

    架構：
        Input → FeatureEmbedding → PositionalEncoding
              → TransformerEncoder (Pre-LN, N layers, + ALiBi recency bias)
              → 取最後有效時間步的隱藏狀態
              → action_head   → (n_action_classes,)
              → point_head    → (n_point_classes,)
              → winner_head   → (1,)

    ALiBi recency bias：
        attention_score[i, j] += -slope × max(0, i-j)
        越舊的 key (j 越小) 在 query i 的眼中扣分越多，
        讓 attention 自然偏向近端拍次。slope 為可學習參數。
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        self.n_heads = cfg.n_heads

        self.feature_embedding = FeatureEmbedding(cfg)
        self.pos_encoding = PositionalEncoding(d_model, max_len=cfg.max_seq_len)

        # Phase B: match-level player fingerprint projection
        # input: (B, L, 2*fp_dim) → output: (B, L, d_model)，加到 stroke embedding
        self.use_match_fingerprint = bool(getattr(cfg, "use_match_fingerprint", False))
        if self.use_match_fingerprint:
            fp_dim = int(cfg.fingerprint_dim)
            self.fp_proj = nn.Sequential(
                nn.LayerNorm(2 * fp_dim),
                nn.Linear(2 * fp_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.fp_proj = None

        # ALiBi recency bias：每個 attention head 各有一個獨立 slope
        # 初始化為不同大小的正值（參考 ALiBi 原論文的 geometric 初始化）
        slopes_init = torch.tensor(
            [1 / (2 ** (8 * i / cfg.n_heads)) for i in range(1, cfg.n_heads + 1)],
            dtype=torch.float,
        )
        self.recency_slopes = nn.Parameter(slopes_init)  # (n_heads,)

        # 手寫 encoder：繞開 nn.TransformerEncoder 的 fused kernel 和 backend
        # dispatch，避免 Flash/MemoryEfficient attention 在 eval 模式下遇到
        # 3D float mask (ALiBi + padding) 產生全 NaN 的 PyTorch 暗坑。
        self.encoder = ManualEncoder(
            d_model=d_model,
            n_heads=cfg.n_heads,
            dim_ff=d_model * 4,
            dropout=cfg.dropout,
            n_layers=cfg.n_layers,
        )
        self.norm = nn.LayerNorm(d_model)

        # Action 分類頭 (19 類)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model * 2, cfg.n_action_classes),
        )

        # Point 分類頭 (10 類)
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model * 2, cfg.n_point_classes),
        )

        # Winner 分類頭 (二分類)
        self.winner_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, 1),
        )
        self.winner_short_expert = (
            WinnerShortPrefixExpert(cfg)
            if bool(getattr(cfg, "use_winner_short_expert", False)) else None
        )

        # Phase C: Action category aux head (5 類: Zero/Attack/Control/Defensive/Serve)
        self.use_action_category = bool(getattr(cfg, "use_action_category", False))
        if self.use_action_category:
            self.action_category_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d_model, cfg.n_action_category_classes),
            )
        else:
            self.action_category_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_recency_bias(self, max_len, device):
        """
        ALiBi recency bias matrix: (n_heads, max_len, max_len)
        bias[h, i, j] = -slopes[h] × max(0, i-j)

        query position i attending to key position j：
          - 若 j < i（舊拍），扣分與距離成正比
          - 若 j >= i（同位置或未來 padding），不扣分
        """
        pos = torch.arange(max_len, device=device, dtype=torch.float)
        # (max_len, max_len)：query - key distance，負值 clamp 為 0
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)
        # slopes: (n_heads,) → (n_heads, 1, 1)
        slopes = self.recency_slopes.abs().unsqueeze(-1).unsqueeze(-1)
        # (n_heads, max_len, max_len)
        return -(slopes * dist)

    @staticmethod
    def _nan_probe(name, t, seq_lens=None):
        """診斷用：只在 eval 且 t 有 NaN/Inf 時列印"""
        import os
        if os.environ.get("NAN_PROBE", "0") != "1":
            return
        if not torch.is_tensor(t):
            return
        bad = (~torch.isfinite(t)).any().item()
        if bad:
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            print(f"  [NaN-PROBE] {name}: shape={tuple(t.shape)} "
                  f"nan={n_nan} inf={n_inf} "
                  f"min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else 'all-bad'} "
                  f"max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else 'all-bad'}")
            if seq_lens is not None:
                print(f"  [NaN-PROBE] seq_lens: min={seq_lens.min().item()} max={seq_lens.max().item()}")

    def forward(self, x, seq_lens, fingerprint=None):
        """
        Args:
            x: (batch, max_seq_len, n_features)
            seq_lens: (batch,)
            fingerprint: 可選 (batch, max_seq_len, 2*fp_dim) — match-level
                player fingerprint。當 cfg.use_match_fingerprint=True 才需提供。
        Returns:
            action_logits: (batch, n_action_classes)
            point_logits:  (batch, n_point_classes)
            winner_logit:  (batch,)
        """
        batch_size = x.size(0)
        max_len = x.size(1)
        # 防禦：seq_lens 最少為 1，避免整個 query row 被 mask 造成 softmax NaN
        seq_lens = seq_lens.clamp(min=1)
        self._nan_probe("00_input_x", x, seq_lens)

        # 組合 ALiBi recency bias + padding mask 成單一 float mask。
        # 用大的「有限」負數代替 -inf：PyTorch eval 模式的 fast-path attention
        # kernel 遇到 -inf row 會 softmax → NaN，換成 -1e9 時即使整 row 被遮罩
        # 也只會得到接近 uniform 的分布（數值無意義但 finite，不會外洩 NaN）。
        NEG_LARGE = -1e9
        alibi = self._build_recency_bias(max_len, x.device)  # (n_heads, L, L)
        # 擴展成 (batch, n_heads, L, L)
        mask = alibi.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        # 對 key position j >= seq_len 的欄位填 NEG_LARGE (等同 key_padding_mask)
        pad_key = torch.arange(max_len, device=x.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        mask = mask.masked_fill(
            pad_key.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, L)
            NEG_LARGE,
        )
        # 攤平成 (batch * n_heads, L, L)，符合 nn.TransformerEncoder 的 mask 規範
        mask = mask.reshape(batch_size * self.n_heads, max_len, max_len)

        self._nan_probe("01_mask", mask)
        self._nan_probe("02_recency_slopes", self.recency_slopes)

        h = self.feature_embedding(x)
        self._nan_probe("03_after_feat_embed", h)

        if self.fp_proj is not None and fingerprint is not None:
            h = h + self.fp_proj(fingerprint)
            self._nan_probe("03b_after_fp_add", h)

        h = self.pos_encoding(h)
        self._nan_probe("04_after_posenc", h)

        # 手寫 encoder，一律走純 PyTorch 運算，無 backend dispatch
        h = self.encoder(h, attn_mask=mask)
        self._nan_probe("05_after_encoder", h)

        h = self.norm(h)
        self._nan_probe("06_after_layernorm", h)

        # 取每個序列最後一個有效時間步
        last_idx = (seq_lens - 1).clamp(min=0)
        last_h = h[torch.arange(batch_size, device=x.device), last_idx]
        self._nan_probe("07_last_h", last_h)

        action_logits = self.action_head(last_h)
        point_logits = self.point_head(last_h)
        winner_logit = self.winner_head(last_h).squeeze(-1)
        if self.winner_short_expert is not None:
            winner_logit = self.winner_short_expert(last_h, seq_lens, winner_logit)
        category_logits = (
            self.action_category_head(last_h)
            if self.action_category_head is not None else None
        )
        self._nan_probe("08_action_logits", action_logits)
        self._nan_probe("09_point_logits", point_logits)
        self._nan_probe("10_winner_logit", winner_logit)

        return action_logits, point_logits, winner_logit, category_logits


# ============================================================
# ShuttleNet-inspired 架構（Phase 1）
# ------------------------------------------------------------
# 參考論文：Hsu et al. 2026, "A New Table Tennis Match Stroke
# Forecasting Method Using Transformer-Based Deep Neural Networks"
# (MJSSM March 2026)
#
# Phase 1 包含：
#   ✅ 雙 stroke embedding（technique 視角 e_t / area 視角 e_a）
#   ✅ Rally Extractor（一個 encoder 跑整序列，看 e_a）
#   ✅ Player Extractor（共用 encoder，奇偶拆分跑 e_t）
#   ✅ Position-Aware Gated Fusion（融合 rally / player A / player B 三流）
#   ✅ 三個 head（action / point / winner）
#   ✅ 沿用現有 ManualEncoder（NaN-safe）作為 backbone
#   ❌ 暫不做 Type-Area Attention 與 encoder-decoder（Phase 2）
#   ❌ 暫不做 k=2/k=4 sliding window 調整（Phase 3）
# ============================================================


class StrokeEmbedding(nn.Module):
    """
    雙視角 stroke embedding（ShuttleNet 改編）：

      e_t (technique view) = projection( actionId, strikeId
                                       + 共用：spinId, strengthId, sex, handId,
                                               playerHand, receiverHand, handPair,
                                               continuous (score / progress / handConf) )

      e_a (area view)      = projection( pointId, pointId_norm, positionId, positionId_norm
                                       + 共用 (同上) )

    共用部分使用同一份 nn.Embedding 實例（不重複學）；視角差異只在最後的 Linear
    projection（讓模型自行決定如何組合）。
    """

    # 視角專屬類別特徵（其餘是共用）
    TECHNIQUE_ONLY = ["actionId", "strikeId"]
    AREA_ONLY = ["pointId", "pointId_norm", "positionId", "positionId_norm"]
    # Player identity 屬於兩視角共用（影響打法 + 站位偏好），且要做 rally-level mask
    PLAYER_ID_FEATURES = ("gamePlayerId", "gamePlayerOtherId")

    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.embed_dim
        d_model = cfg.d_model
        self.feature_names = cfg.all_features

        # 所有類別特徵共用一份 embedding
        self.embeddings = nn.ModuleDict()
        for feat_name, n_classes in cfg.categorical_features.items():
            self.embeddings[feat_name] = nn.Embedding(
                n_classes + 2, embed_dim, padding_idx=0
            )

        # 分組：哪些特徵屬於 technique 專屬 / area 專屬 / 共用
        all_cat = list(cfg.categorical_features.keys())
        self.tech_only_cat = [f for f in self.TECHNIQUE_ONLY if f in cfg.categorical_features]
        self.area_only_cat = [f for f in self.AREA_ONLY if f in cfg.categorical_features]
        self.shared_cat = [
            f for f in all_cat
            if f not in self.tech_only_cat and f not in self.area_only_cat
        ]

        # 連續特徵：避免重複（如果某 feature 同時被列為類別特徵就不算連續）
        self._pure_continuous = [
            f for f in cfg.continuous_features + cfg.engineered_features
            if f not in cfg.categorical_features
        ]
        n_continuous = len(self._pure_continuous)
        self.continuous_ln = nn.LayerNorm(n_continuous)
        self.continuous_proj = nn.Linear(n_continuous, embed_dim)

        # 兩個視角各自的投射：把該視角看到的 embedding concat 後投射到 d_model
        n_shared = len(self.shared_cat)
        n_tech_only = len(self.tech_only_cat)
        n_area_only = len(self.area_only_cat)
        # +1 (×embed_dim) 是因為連續特徵也佔一格 embed_dim
        tech_total_dim = (n_shared + n_tech_only + 1) * embed_dim
        area_total_dim = (n_shared + n_area_only + 1) * embed_dim

        def make_proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            )

        self.tech_projection = make_proj(tech_total_dim)
        self.area_projection = make_proj(area_total_dim)

        # 預計算索引
        self._cont_indices = [
            self.feature_names.index(f) for f in self._pure_continuous
        ]
        self._cat_info = {
            feat_name: (self.feature_names.index(feat_name), n_classes)
            for feat_name, n_classes in cfg.categorical_features.items()
        }

        # Phase A: rally-level random masking of player ID
        self.player_id_dropout = float(getattr(cfg, "player_id_dropout", 0.0))
        self._player_id_set = set(self.PLAYER_ID_FEATURES) & set(self._cat_info.keys())

    def _embed_cat_list(self, x, feat_names, player_drop_mask=None):
        """從 x 提取指定的類別特徵並查表，返回 list of (B, L, embed_dim)。

        player_drop_mask: (B,) bool — 對應 True 的 row，把 PLAYER_ID_FEATURES 的
        值統一改成 0（unknown），等同丟掉該 rally 的選手身份資訊。
        """
        outs = []
        B = x.size(0)
        for f in feat_names:
            idx, n_cls = self._cat_info[f]
            vals = x[:, :, idx].long().clamp(0, n_cls + 1)
            if player_drop_mask is not None and f in self._player_id_set:
                vals = vals.masked_fill(player_drop_mask.view(B, 1), 0)
            outs.append(self.embeddings[f](vals))
        return outs

    def forward(self, x):
        """
        Args:
            x: (B, L, n_features)
        Returns:
            e_t: (B, L, d_model) 技術視角 embedding
            e_a: (B, L, d_model) 區域視角 embedding
        """
        B = x.size(0)
        # 訓練時對「整個 rally」隨機把 player ID mask 成 unknown
        if (self.training and self.player_id_dropout > 0.0
                and len(self._player_id_set) > 0):
            drop = torch.rand(B, device=x.device) < self.player_id_dropout
        else:
            drop = None

        # 共用部分（兩個視角都看到）— player ID 在這裡，套用 drop
        shared_embs = self._embed_cat_list(x, self.shared_cat, player_drop_mask=drop)
        cont_feats = x[:, :, self._cont_indices]
        cont_normed = self.continuous_ln(cont_feats)
        cont_emb = self.continuous_proj(cont_normed)
        shared_embs.append(cont_emb)

        # 視角專屬（不含 player ID，drop 對它們無作用）
        tech_embs = self._embed_cat_list(x, self.tech_only_cat)
        area_embs = self._embed_cat_list(x, self.area_only_cat)

        # 拼接 + 投射
        e_t = self.tech_projection(torch.cat(shared_embs + tech_embs, dim=-1))
        e_a = self.area_projection(torch.cat(shared_embs + area_embs, dim=-1))
        return e_t, e_a


class GatedFusion(nn.Module):
    """
    Position-Aware Gated Fusion（ShuttleNet 改編）。

        z = β_A · α_A · h̃_A  +  β_B · α_B · h̃_B  +  β_R · α_R · h̃_R

    其中：
      h̃_X = tanh(W_X h_X)                                 — 各流的非線性投射
      α_X = sigmoid(W_α^X · concat([h_A, h_B, h_R]))       — 從整體上下文算 gate
      β_X — 每流一個可學習純量

    註：原論文公式外層還有 σ()，但會把 z 限制在 [0, 1] 以致下游 head 的 logit
    動態範圍受限；常見 gated fusion 實作不加外層 σ，這裡也省略。

    對於 player B 不存在的樣本（rally 只有一拍）會以 mask_b=False 把該流關閉。
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 各流的 tanh 投射 W_X
        self.proj_a = nn.Linear(d_model, d_model)
        self.proj_b = nn.Linear(d_model, d_model)
        self.proj_r = nn.Linear(d_model, d_model)
        # 從 concat([h_a, h_b, h_r]) 算 gate
        self.gate_a = nn.Linear(3 * d_model, d_model)
        self.gate_b = nn.Linear(3 * d_model, d_model)
        self.gate_r = nn.Linear(3 * d_model, d_model)
        # 可學習純量 β
        self.beta_a = nn.Parameter(torch.ones(1))
        self.beta_b = nn.Parameter(torch.ones(1))
        self.beta_r = nn.Parameter(torch.ones(1))

    def forward(self, h_a, h_b, h_r, mask_a=None, mask_b=None):
        """
        Args:
            h_a, h_b, h_r: (B, d_model) 三個流的最後位置表示
            mask_a, mask_b: 可選 (B,) bool，True 代表該流有效
        Returns:
            z: (B, d_model) 融合後的表示
        """
        # tanh 投射
        h_a_t = torch.tanh(self.proj_a(h_a))
        h_b_t = torch.tanh(self.proj_b(h_b))
        h_r_t = torch.tanh(self.proj_r(h_r))

        # gate from full context
        ctx = torch.cat([h_a, h_b, h_r], dim=-1)
        alpha_a = torch.sigmoid(self.gate_a(ctx))
        alpha_b = torch.sigmoid(self.gate_b(ctx))
        alpha_r = torch.sigmoid(self.gate_r(ctx))

        # 對於不存在的流（如 rally 只有 1 拍 → 沒有 player B），把 gate 強制歸零
        if mask_a is not None:
            alpha_a = alpha_a * mask_a.float().unsqueeze(-1)
        if mask_b is not None:
            alpha_b = alpha_b * mask_b.float().unsqueeze(-1)

        z = (
            self.beta_a * alpha_a * h_a_t
            + self.beta_b * alpha_b * h_b_t
            + self.beta_r * alpha_r * h_r_t
        )
        return z


# ============================================================
# Phase 2 + Phase 3: TAA decoder（Type-Area Attention + sliding window）
# ------------------------------------------------------------
# Phase 2：在 encoder 之上加上一個 single-step decoder
#   - 兩個 learnable task query: q_t (technique) 與 q_a (area)
#   - Cross-attention：q_t → player encoders (e_t-based)
#                      q_a → rally encoder (e_a-based)
#   - Type-Area Attention (TAA)：q_t 與 q_a 相互交換資訊
#
# Phase 3：在 cross-attention 中加入 hard sliding window mask
#   論文 k=2 / k=4 是「看最近 k 拍預測下一拍」的 lookback 視窗。改成硬遮罩
#   而非 ALiBi soft bias，邊界更明確。
#     - q_t (action): 從 player A / player B 各看最後 1 拍 → 共 2 拍
#     - q_a (point):  從 rally encoder 看最後 4 拍
# ============================================================


class TAACrossAttention(nn.Module):
    """
    單 query cross-attention（query 長度恆為 1），使用 hard sliding window
    mask 限制只看最後 window_k 個有效 key。

    輸入：
        query:        (B, 1, D)
        keys_values:  (B, L, D)
        kv_seq_lens:  (B,) — keys/values 的有效長度（其餘為 padding）

    Sliding window mask：
        對每個 batch，只允許 attention 看到 kv 的最後 window_k 個有效位置；
        更舊的位置與 padding 位置都填 -1e9 強制 softmax 後 ≈0。

        例如 kv_seq_lens=5, window_k=2 → 只看 index 3, 4。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, window_k: int):
        super().__init__()
        assert d_model % n_heads == 0
        assert window_k >= 1
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_k = int(window_k)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, keys_values, kv_seq_lens):
        """
        query:        (B, 1, D)
        keys_values:  (B, L, D)
        kv_seq_lens:  (B,)  >= 1 期望（呼叫端負責 clamp，否則 softmax NaN）
        return:       (B, 1, D)
        """
        B, L, D = keys_values.shape
        H, Dh = self.n_heads, self.d_head
        device = keys_values.device

        q = self.q_proj(query).view(B, 1, H, Dh).transpose(1, 2)         # (B, H, 1, Dh)
        k = self.k_proj(keys_values).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        v = self.v_proj(keys_values).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)    # (B, H, 1, L)

        # Hard sliding window mask + padding mask
        # 「最後一個有效拍」index = kv_seq_lens - 1
        # 允許看到的範圍：dist_from_last ∈ [0, window_k - 1]
        pos = torch.arange(L, device=device).view(1, L)                  # (1, L)
        last_pos = (kv_seq_lens - 1).view(B, 1)                          # (B, 1)
        dist_from_last = last_pos - pos                                  # (B, L)
        too_old = dist_from_last >= self.window_k                        # 距離 >= k 的舊拍
        is_padding = pos >= kv_seq_lens.view(B, 1)                       # padding 位置
        # too_old 已含 padding 的情況？不一定 → 兩者 OR
        block = (too_old | is_padding).view(B, 1, 1, L)                  # (B, 1, 1, L)
        # AMP-safe negative：fp16 max abs ≈ 65504，固定 -1e9 會 overflow
        # 報「value cannot be converted to type c10::Half」。改用 dtype.min
        # 自動配合 fp16 / bf16 / fp32，行為與 -∞ 等價（exp(min) = 0）。
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(block, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)                                      # (B, H, 1, Dh)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.out_proj(out)


class TAADecoder(nn.Module):
    """
    Type-Area Attention Decoder（Phase 2 + Phase 3）

    架構：
        q_t (technique query)  ┐                                ┌→ z_t → action_head
                              ├→ Cross-attn (Phase 2)          │     winner_head
                              ↓                                  │     (參與 winner)
                          TAA exchange ┐
                              ↑                                  │
                              ├→ Cross-attn (Phase 2)          │
        q_a (area query)       ┘                                └→ z_a → point_head

    細節：
      - q_t cross-attends to h_pa 與 h_pb（player encoders, e_t-based, k=1+1=2 拍）
      - q_a cross-attends to h_rally（rally encoder, e_a-based, k=4 拍）
      - TAA exchange：q_t/q_a 各做一次 gated linear transform 從對方吸收資訊
      - 兩個視角各自接 FFN + Pre-LN
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        # Learnable task queries
        self.q_t = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_a = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Phase 3: task-specific hard sliding window
        # 論文 k 是「看最近 k 拍預測下一拍」的 lookback 視窗。
        #   action (technique): k=2 拍 → 從 player A / B 各看最後 1 拍 (1+1=2)
        #   point  (area):      k=4 拍 → 從 rally encoder 看最後 4 拍
        # 把 player 平均拆給 A / B，因為他們交替出拍，各看 1 拍正好包含「上一拍對手 + 自己上一拍」。
        WINDOW_PLAYER = 1   # 每個 player encoder 看最後 1 拍 → 加總正好 2 拍
        WINDOW_RALLY = 4    # rally encoder 看最後 4 拍

        self.cross_t_pa = TAACrossAttention(d_model, n_heads, dropout, WINDOW_PLAYER)
        self.cross_t_pb = TAACrossAttention(d_model, n_heads, dropout, WINDOW_PLAYER)
        self.cross_a_r = TAACrossAttention(d_model, n_heads, dropout, WINDOW_RALLY)

        # TAA exchange: gated linear from one view to the other
        self.taa_t_proj = nn.Linear(d_model, d_model)
        self.taa_a_proj = nn.Linear(d_model, d_model)
        self.taa_t_gate = nn.Linear(2 * d_model, d_model)
        self.taa_a_gate = nn.Linear(2 * d_model, d_model)

        # Pre-LN around residuals
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_t2 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

        # FFN
        def make_ffn():
            return nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )

        self.ffn_t = make_ffn()
        self.ffn_a = make_ffn()

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_rally, h_pa, h_pb, sl_rally, sl_pa, sl_pb, has_b):
        """
        Args:
            h_rally: (B, L_r, D)  rally encoder output (area-based)
            h_pa:    (B, L_a, D)  player A encoder output (technique-based)
            h_pb:    (B, L_b, D)  player B encoder output (technique-based)
            sl_rally, sl_pa, sl_pb: (B,) 各自有效長度
            has_b:   (B,) bool — True 表示 player B 存在（sl_pb >= 1）

        Returns:
            z_t: (B, D) 技術視角輸出（action / winner 用）
            z_a: (B, D) 區域視角輸出（point 用）
        """
        B = h_rally.size(0)
        device = h_rally.device

        # ---- Expand learnable queries ----
        q_t = self.q_t.expand(B, -1, -1)  # (B, 1, D)
        q_a = self.q_a.expand(B, -1, -1)  # (B, 1, D)

        # ---- Phase 2: cross-attention ----
        # q_t → player A
        ca_pa = self.cross_t_pa(q_t, h_pa, sl_pa)                       # (B, 1, D)

        # q_t → player B（防呆：sl_pb 可能為 0，先 clamp 再用 has_b 歸零結果）
        sl_pb_safe = sl_pb.clamp(min=1)
        ca_pb = self.cross_t_pb(q_t, h_pb, sl_pb_safe)                  # (B, 1, D)
        ca_pb = ca_pb * has_b.float().view(B, 1, 1)

        # 平均兩個 player stream（單拍 rally 時只有 A）
        n_streams = 1.0 + has_b.float()                                 # (B,)
        ca_player = (ca_pa + ca_pb) / n_streams.view(B, 1, 1)

        q_t_mid = self.norm_t1(q_t + self.dropout(ca_player))

        # q_a → rally
        ca_r = self.cross_a_r(q_a, h_rally, sl_rally)                   # (B, 1, D)
        q_a_mid = self.norm_a1(q_a + self.dropout(ca_r))

        # ---- TAA exchange: 兩個視角互相交換 ----
        # q_t 從 q_a_mid 吸收
        ctx_t = torch.cat([q_t_mid, q_a_mid], dim=-1)                   # (B, 1, 2D)
        gate_t = torch.sigmoid(self.taa_t_gate(ctx_t))                  # (B, 1, D)
        delta_t = self.taa_t_proj(q_a_mid)                              # (B, 1, D)
        q_t_taa = q_t_mid + gate_t * delta_t

        # q_a 從 q_t_mid 吸收
        ctx_a = torch.cat([q_a_mid, q_t_mid], dim=-1)
        gate_a = torch.sigmoid(self.taa_a_gate(ctx_a))
        delta_a = self.taa_a_proj(q_t_mid)
        q_a_taa = q_a_mid + gate_a * delta_a

        # ---- FFN（Pre-LN） ----
        q_t_out = q_t_taa + self.ffn_t(self.norm_t2(q_t_taa))
        q_a_out = q_a_taa + self.ffn_a(self.norm_a2(q_a_taa))

        return q_t_out.squeeze(1), q_a_out.squeeze(1)


class ShuttleNetModel(nn.Module):
    """
    ShuttleNet-inspired 多任務模型（Phase 1：encoder-only + 雙視角 + gated fusion）。

    架構：
        Input → StrokeEmbedding → (e_t, e_a) （兩個視角）
                ↓ + PositionalEncoding
        ┌── Rally Extractor: ManualEncoder(e_a, 整序列) ─→ h_r [last]
        ├── Player A Extractor: ManualEncoder(e_t[:, 0::2], 奇拍) ─→ h_a [last]
        └── Player B Extractor: ManualEncoder(e_t[:, 1::2], 偶拍) ─→ h_b [last]
                                       ↓
                            Position-Aware Gated Fusion → z
                                       ↓
                action_head(z), point_head(z), winner_head(z)

    說明：
      - 論文用 area embedding 給 rally extractor、technique embedding 給 player extractor，
        因為 rally 著重「球落點軌跡」而 player 著重「個人打法」。
      - Player A = 發球方（奇數拍 = strikeNumber 1, 3, 5, ...，序列 index 0, 2, 4, ...）
      - Player B = 接球方（偶數拍 = strikeNumber 2, 4, ...，序列 index 1, 3, ...）
      - Player A 與 Player B 共用同一個 encoder（玩家角色對稱、更省參數、更省資料）。
      - 沿用 MultiTaskTransformer 的 ManualEncoder + ALiBi recency bias，避開
        PyTorch fused attention 在 eval 模式遇 3D float mask 全 NaN 的暗坑。
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.max_seq_len = cfg.max_seq_len
        # Phase 2/3 開關：True 用 TAADecoder（雙視角 + TAA + sliding window），
        # False 用 Phase 1 的 last-position pooling + GatedFusion
        self.use_taa_decoder = bool(getattr(cfg, "use_taa_decoder", False))

        # 雙視角 embedding
        self.stroke_embedding = StrokeEmbedding(cfg)
        # 位置編碼共用
        self.pos_encoding = PositionalEncoding(d_model, max_len=cfg.max_seq_len)

        # Phase B: match-level player fingerprint projection（兩個視角都加）
        self.use_match_fingerprint = bool(getattr(cfg, "use_match_fingerprint", False))
        if self.use_match_fingerprint:
            fp_dim = int(cfg.fingerprint_dim)
            self.fp_proj = nn.Sequential(
                nn.LayerNorm(2 * fp_dim),
                nn.Linear(2 * fp_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.fp_proj = None

        def make_encoder():
            return ManualEncoder(
                d_model=d_model,
                n_heads=cfg.n_heads,
                dim_ff=d_model * 4,
                dropout=cfg.dropout,
                n_layers=cfg.n_layers,
            )

        # Rally Extractor 看 e_a；Player A/B 共用一個 encoder 看 e_t
        self.rally_encoder = make_encoder()
        self.player_encoder = make_encoder()

        # 兩組獨立的 ALiBi slopes（rally 觀察整 rally；player 觀察玩家自己的拍序）
        def init_slopes():
            return nn.Parameter(torch.tensor(
                [1 / (2 ** (8 * i / cfg.n_heads)) for i in range(1, cfg.n_heads + 1)],
                dtype=torch.float,
            ))

        self.slopes_rally = init_slopes()
        self.slopes_player = init_slopes()

        # 各 stream 的最後 LayerNorm
        self.norm_rally = nn.LayerNorm(d_model)
        self.norm_player = nn.LayerNorm(d_model)

        # Phase 1：Position-Aware Gated Fusion；Phase 2/3：TAA Decoder
        if self.use_taa_decoder:
            self.decoder = TAADecoder(d_model, cfg.n_heads, cfg.dropout)
        else:
            self.fusion = GatedFusion(d_model)

        # 三個 head（與 MultiTaskTransformer 同設計）
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model * 2, cfg.n_action_classes),
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model * 2, cfg.n_point_classes),
        )
        # v38 任務 A：action long-view 路徑。z_t（TAA decoder k=2 hard window）
        # 看「下一拍最近 context」，再用 masked-mean(h_rally) → action_pool_proj
        # 加一條「rally 全局 pattern」訊號，最後用 action_fusion(Linear) 合併。
        # False 時 forward 維持原 v37 行為（action_logits = action_head(z_t)）。
        self.use_action_long_view = bool(getattr(cfg, "use_action_long_view", False))
        if self.use_action_long_view:
            self.action_pool_proj = nn.Linear(d_model, d_model)
            self.action_fusion = nn.Linear(2 * d_model, d_model)
        # v39 任務 A1：point long-view 路徑（鏡像 action long-view）。
        # z_a 來自 TAA decoder 的 k=4 hard window，pointId 需要看到 rally 全局
        # 的空間 transition / cumulative count，long-view 補上這塊。
        self.use_point_long_view = bool(getattr(cfg, "use_point_long_view", False))
        if self.use_point_long_view:
            self.point_pool_proj = nn.Linear(d_model, d_model)
            self.point_fusion = nn.Linear(2 * d_model, d_model)
        # v37 任務 2：給 winner head 一條獨立的全 rally 長視野路徑（masked mean
        # over h_rally → Linear → winner_head）。z_t / z_a 是 hard sliding window
        # (k=2/k=4) 產生，視野對「整 rally 勝負」太短。
        self.use_winner_long_view = bool(getattr(cfg, "use_winner_long_view", False))
        if self.use_winner_long_view:
            self.winner_pool_proj = nn.Linear(d_model, d_model)
        self.winner_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, 1),
        )
        self.winner_short_expert = (
            WinnerShortPrefixExpert(cfg)
            if bool(getattr(cfg, "use_winner_short_expert", False)) else None
        )
        # v38 任務 B：action 短前綴專家（鏡像 winner_short_expert）。
        self.action_short_expert = (
            ActionShortPrefixExpert(cfg)
            if bool(getattr(cfg, "use_action_short_expert", False)) else None
        )
        # v39 任務 A2：point 短前綴專家（鏡像 action_short_expert）。
        self.point_short_expert = (
            PointShortPrefixExpert(cfg)
            if bool(getattr(cfg, "use_point_short_expert", False)) else None
        )
        # v41：fingerprint → action 直接路徑（conditional-shift domain adaptation）。
        # 把「下一拍 striker 的 fingerprint action 直方圖」直接注入 action logit
        # bias，繞過 encoder 對這個選手習慣強信號的稀釋。需有 fingerprint 才啟用。
        self.n_action_classes = int(cfg.n_action_classes)
        self.use_fp_action_prior = (
            bool(getattr(cfg, "use_fp_action_prior", False))
            and self.use_match_fingerprint
        )
        self.fp_action_prior = (
            FingerprintActionPrior(self.n_action_classes, cfg.dropout)
            if self.use_fp_action_prior else None
        )

        # Phase C: Action category aux head（與 action_head 共用同一個輸入表示）
        self.use_action_category = bool(getattr(cfg, "use_action_category", False))
        if self.use_action_category:
            self.action_category_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d_model, cfg.n_action_category_classes),
            )
        else:
            self.action_category_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _build_recency_bias(slopes, max_len, device):
        """ALiBi recency bias matrix: (n_heads, max_len, max_len)"""
        pos = torch.arange(max_len, device=device, dtype=torch.float)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)
        s = slopes.abs().unsqueeze(-1).unsqueeze(-1)
        return -(s * dist)

    def _build_mask(self, slopes, max_len, seq_lens, device):
        """組合 ALiBi recency bias + key padding mask 成 (B*H, L, L) float mask"""
        B = seq_lens.size(0)
        H = self.n_heads
        NEG_LARGE = -1e9
        alibi = self._build_recency_bias(slopes, max_len, device)  # (H, L, L)
        mask = alibi.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B, H, L, L)
        pad_key = torch.arange(max_len, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        mask = mask.masked_fill(pad_key.unsqueeze(1).unsqueeze(2), NEG_LARGE)
        return mask.reshape(B * H, max_len, max_len)

    @staticmethod
    def _gather_last(h, seq_lens):
        """從 h: (B, L, D) 取每個序列最後一個有效時間步"""
        B = h.size(0)
        last_idx = (seq_lens - 1).clamp(min=0)
        return h[torch.arange(B, device=h.device), last_idx]

    @staticmethod
    def _split_player_streams(e, seq_lens):
        """
        將 e: (B, L, D) 依 stroke index 奇偶切成 player A / player B 子序列。
          - 序列 index 0, 2, 4, ... → strikeNumber 1, 3, 5, ... → server (Player A)
          - 序列 index 1, 3, 5, ... → strikeNumber 2, 4, 6, ... → receiver (Player B)

        Returns:
            e_a:  (B, L_a, D)  player A 子序列（已 padding 到等長）
            e_b:  (B, L_b, D)  player B 子序列
            sl_a: (B,)  player A 有效拍數 = ceil(seq_lens/2)
            sl_b: (B,)  player B 有效拍數 = floor(seq_lens/2)
        """
        e_a = e[:, 0::2, :]   # (B, ceil(L/2), D)
        e_b = e[:, 1::2, :]   # (B, floor(L/2), D)
        sl_a = (seq_lens + 1) // 2
        sl_b = seq_lens // 2
        return e_a, e_b, sl_a, sl_b

    def forward(self, x, seq_lens, fingerprint=None):
        """
        Args:
            x: (B, L, n_features)
            seq_lens: (B,)
            fingerprint: 可選 (B, L, 2*fp_dim) — match-level player fingerprint。
                只在 cfg.use_match_fingerprint=True 時使用。同一條 fp_emb
                同時加到 e_t 與 e_a 兩個視角（同 player 在 match 內的指紋
                對技術 / 區域兩種預測都有資訊量）。
        Returns:
            action_logits: (B, n_action_classes)
            point_logits:  (B, n_point_classes)
            winner_logit:  (B,)
        """
        B = x.size(0)
        L = x.size(1)
        device = x.device
        # 防禦：seq_lens 至少為 1，避免整 row 被 mask 造成 softmax NaN
        seq_lens = seq_lens.clamp(min=1)

        # ---- 1. 雙視角 stroke embedding + 位置編碼 ----
        e_t, e_a = self.stroke_embedding(x)            # 各 (B, L, d_model)

        # Match-level fingerprint：同一向量加到兩個視角（在 pos_encoding 之前
        # 拼進去比較像「靜態身份標記」，模型可自行決定要不要 attend）。
        if self.fp_proj is not None and fingerprint is not None:
            fp_emb = self.fp_proj(fingerprint)         # (B, L, d_model)
            e_t = e_t + fp_emb
            e_a = e_a + fp_emb

        e_t = self.pos_encoding(e_t)
        e_a = self.pos_encoding(e_a)

        # ---- 2. Rally Extractor（看 area 視角，整序列） ----
        mask_rally = self._build_mask(self.slopes_rally, L, seq_lens, device)
        h_rally = self.rally_encoder(e_a, attn_mask=mask_rally)
        h_rally = self.norm_rally(h_rally)

        # v37 任務 2 + v38 任務 A + v39 任務 A1：winner / action / point head 共用
        # 全 rally 長視野 representation。masked mean over valid timesteps（不含
        # padding），再各自過 Linear 投影。三者只要任一啟用就算 h_rally_mean。
        _need_long_view = (
            self.use_winner_long_view
            or self.use_action_long_view
            or self.use_point_long_view
        )
        if _need_long_view:
            mask_lv = (
                torch.arange(L, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
            ).float().unsqueeze(-1)                          # (B, L, 1)
            h_rally_mean = (h_rally * mask_lv).sum(dim=1) / mask_lv.sum(dim=1).clamp(min=1.0)
        else:
            h_rally_mean = None
        winner_long_view = (
            self.winner_pool_proj(h_rally_mean) if self.use_winner_long_view else None
        )
        action_long_view = (
            self.action_pool_proj(h_rally_mean) if self.use_action_long_view else None
        )
        point_long_view = (
            self.point_pool_proj(h_rally_mean) if self.use_point_long_view else None
        )

        # v41：算 fingerprint→action bias（下一拍 striker 的 action 直方圖直接注入）。
        # 下一拍 striker = 最後 input stroke 的對手（gamePlayerId 每拍交替）→ 取 other_fp
        # （fingerprint 後半段）的 action 直方圖；信心度用 other 的 log_n_other（最後一格）。
        fp_action_bias = None
        if self.fp_action_prior is not None and fingerprint is not None:
            last_idx = (seq_lens - 1).clamp(min=0)                       # (B,)
            fp_last = fingerprint[torch.arange(B, device=device), last_idx]  # (B, 2*fp_dim)
            fp_inner = fingerprint.size(-1) // 2                         # = fp_dim (44)
            na = self.n_action_classes
            next_hist = fp_last[:, fp_inner: fp_inner + na]              # (B, na) 對手 action 直方圖
            next_conf = fp_last[:, 2 * fp_inner - 1: 2 * fp_inner]       # (B, 1) 對手 log_n_other
            fp_action_bias = self.fp_action_prior(next_hist, next_conf)

        # ---- 3. Player Extractor（看 technique 視角，奇偶拆分） ----
        e_t_a, e_t_b, sl_a, sl_b = self._split_player_streams(e_t, seq_lens)
        L_a = e_t_a.size(1)
        L_b = e_t_b.size(1)

        # Player A：sl_a >= 1 永遠成立
        mask_a = self._build_mask(self.slopes_player, L_a, sl_a, device)
        h_pa = self.player_encoder(e_t_a, attn_mask=mask_a)
        h_pa = self.norm_player(h_pa)

        # Player B：當 seq_lens=1 時 sl_b=0（沒接球方），需要保護
        if L_b > 0:
            sl_b_safe = sl_b.clamp(min=1)
            mask_b = self._build_mask(self.slopes_player, L_b, sl_b_safe, device)
            h_pb = self.player_encoder(e_t_b, attn_mask=mask_b)
            h_pb = self.norm_player(h_pb)
            has_b = (sl_b > 0)                                  # (B,) bool
        else:
            # 整個 batch 的 max seq_len 都 = 1（理論上不會發生，因 max_seq_len >= 2）
            h_pb = torch.zeros(B, 1, h_pa.size(-1), device=device)
            sl_b = torch.zeros(B, dtype=torch.long, device=device)
            has_b = torch.zeros(B, dtype=torch.bool, device=device)

        # ---- 4. Decoder / Fusion ----
        if self.use_taa_decoder:
            # Phase 2 + 3：TAA decoder 產出 z_t (技術視角) 與 z_a (區域視角)
            # 兩個視角分別走不同 head
            z_t, z_a = self.decoder(
                h_rally=h_rally,
                h_pa=h_pa,
                h_pb=h_pb,
                sl_rally=seq_lens,
                sl_pa=sl_a,
                sl_pb=sl_b,
                has_b=has_b,
            )
            # v38 任務 A：action_repr = action_fusion(concat[z_t, action_long_view])
            # 同時看「下一拍最近 context」(z_t, k=2 window) 跟「rally 全局 pattern」
            # (masked-mean h_rally)。flag=False 時 action_repr = z_t，與 v37 等價。
            if self.use_action_long_view:
                action_repr = self.action_fusion(
                    torch.cat([z_t, action_long_view], dim=-1)
                )
            else:
                action_repr = z_t
            action_logits = self.action_head(action_repr)
            # v38 任務 B：短前綴專家修正（鏡像 winner short expert）。
            if self.action_short_expert is not None:
                action_logits = self.action_short_expert(
                    action_repr, seq_lens, action_logits
                )
            # v41：fingerprint→action 直接注入（conditional-shift 修正）
            if fp_action_bias is not None:
                action_logits = action_logits + fp_action_bias
            # v39 任務 A1：point_repr = point_fusion(concat[z_a, point_long_view])
            # 同時看「下一拍的 area context」(z_a, k=4 window) 跟「rally 全局空間
            # 轉移」(masked-mean h_rally)。flag=False 時 point_repr = z_a。
            if self.use_point_long_view:
                point_repr = self.point_fusion(
                    torch.cat([z_a, point_long_view], dim=-1)
                )
            else:
                point_repr = z_a
            point_logits = self.point_head(point_repr)
            # v39 任務 A2：point 短前綴專家
            if self.point_short_expert is not None:
                point_logits = self.point_short_expert(
                    point_repr, seq_lens, point_logits
                )
            # v37 任務 2：winner head 可選用長視野路徑（masked-mean h_rally → Linear），
            # 否則 fallback 為兩視角平均（v36 行為）。
            if self.use_winner_long_view:
                winner_repr = winner_long_view
            else:
                winner_repr = 0.5 * (z_t + z_a)
            winner_logit = self.winner_head(winner_repr).squeeze(-1)
            if self.winner_short_expert is not None:
                winner_logit = self.winner_short_expert(
                    winner_repr, seq_lens, winner_logit
                )
            # Phase C: action category aux head 共用 z_t（技術視角，與 action 同源）
            category_logits = (
                self.action_category_head(z_t)
                if self.action_category_head is not None else None
            )
        else:
            # Phase 1：last-position pooling + Position-Aware Gated Fusion
            h_r_last = self._gather_last(h_rally, seq_lens)
            h_a_last = self._gather_last(h_pa, sl_a)
            if L_b > 0:
                h_b_last = self._gather_last(h_pb, sl_b.clamp(min=1))
                h_b_last = h_b_last * has_b.float().unsqueeze(-1)
            else:
                h_b_last = torch.zeros_like(h_a_last)
            z = self.fusion(
                h_a=h_a_last,
                h_b=h_b_last,
                h_r=h_r_last,
                mask_a=None,
                mask_b=has_b,
            )
            # v38 任務 A：action_repr 融合 z 跟 action_long_view（Phase 1 分支也支援）
            if self.use_action_long_view:
                action_repr = self.action_fusion(
                    torch.cat([z, action_long_view], dim=-1)
                )
            else:
                action_repr = z
            action_logits = self.action_head(action_repr)
            # v38 任務 B：action short-prefix expert
            if self.action_short_expert is not None:
                action_logits = self.action_short_expert(
                    action_repr, seq_lens, action_logits
                )
            # v41：fingerprint→action 直接注入（conditional-shift 修正）
            if fp_action_bias is not None:
                action_logits = action_logits + fp_action_bias
            # v39 任務 A1：point_repr 融合 z 跟 point_long_view
            if self.use_point_long_view:
                point_repr = self.point_fusion(
                    torch.cat([z, point_long_view], dim=-1)
                )
            else:
                point_repr = z
            point_logits = self.point_head(point_repr)
            # v39 任務 A2：point short-prefix expert
            if self.point_short_expert is not None:
                point_logits = self.point_short_expert(
                    point_repr, seq_lens, point_logits
                )
            # v37 任務 2：use_winner_long_view=True 時 winner 走長視野路徑
            # （Phase 1 分支也支援，雖然平常不會用到）。
            if self.use_winner_long_view:
                winner_repr = winner_long_view
            else:
                winner_repr = z
            winner_logit = self.winner_head(winner_repr).squeeze(-1)
            if self.winner_short_expert is not None:
                winner_logit = self.winner_short_expert(winner_repr, seq_lens, winner_logit)
            category_logits = (
                self.action_category_head(z)
                if self.action_category_head is not None else None
            )

        return action_logits, point_logits, winner_logit, category_logits
