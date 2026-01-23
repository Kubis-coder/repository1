import os
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.features import standardize_columns, parse_strength, parse_time_to_seconds


class NHL_LSTM(nn.Module):
    """
    Musí přesně odpovídat architektuře v train_model_lstm_defense.py
    """
    def __init__(self, num_dim: int, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        input_dim = num_dim + emb_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_evt):
        e = self.emb(x_evt)               # (B,L,E)
        x = torch.cat([x_num, e], dim=-1) # (B,L,D+E)
        out, _ = self.lstm(x)             # (B,L,H)
        last = out[:, -1, :]              # (B,H)
        return self.head(last).squeeze(-1)


class PredictorAgentDL:
    """
    DL prediktor rizika "goal against within next N events".
    - Načítá model + meta (mean/std, threshold, vocab).
    - Vyrábí stejné featury jako trénink (HOCKEY FEATURES).
    """

    def __init__(
        self,
        model_path: str = "models/nhl_brain_lstm_defense.pt",
        meta_path: str = "models/nhl_brain_lstm_defense_meta.json",
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.meta_path = meta_path

        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[NHL_LSTM] = None

        # meta
        self.seq_len: int = 20
        self.target_window: int = 7
        self.event_vocab: List[str] = [
            "SHOT", "GOAL", "MISS", "BLOCK", "HIT", "TAKEAWAY",
            "GIVEAWAY", "PENALTY", "FACEOFF", "OTHER"
        ]
        self.event2id: Dict[str, int] = {e: i for i, e in enumerate(self.event_vocab)}

        self.numeric_dim: int = 12
        self.numeric_mean: Optional[np.ndarray] = None
        self.numeric_std: Optional[np.ndarray] = None

        # threshold (default + načtení z meta)
        self.threshold: float = 0.5

        self._load()


    def _load(self) -> None:
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta nenalezena: {self.meta_path}")

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.seq_len = int(meta.get("seq_len", 20))
        self.target_window = int(meta.get("target_window", 7))

        self.event_vocab = list(meta.get("event_vocab", self.event_vocab))
        self.event2id = {e: i for i, e in enumerate(self.event_vocab)}

        self.numeric_dim = int(meta.get("numeric_dim", 12))

        mean = meta.get("numeric_mean", None)
        std = meta.get("numeric_std", None)
        if mean is not None and std is not None:
            self.numeric_mean = np.array(mean, dtype=np.float32)
            self.numeric_std = np.array(std, dtype=np.float32)

        # ✅ threshold z meta
        self.threshold = float(meta.get("threshold", 0.5))

        # model arch z meta
        emb_dim = int(meta.get("emb_dim", 16))
        hidden_dim = int(meta.get("hidden_dim", 64))
        num_layers = int(meta.get("num_layers", 2))
        dropout = float(meta.get("dropout", 0.3))

        self.model = NHL_LSTM(
            num_dim=self.numeric_dim,
            vocab_size=len(self.event_vocab),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model nenalezen: {self.model_path}")

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    # =========================
    # FEATURE HELPERS
    # =========================
    def _map_event(self, evt) -> int:
        s = str(evt).upper()
        for k in self.event2id.keys():
            if k != "OTHER" and k in s:
                return self.event2id[k]
        return self.event2id.get("OTHER", 0)

    def _compute_abs_time(self, df: pd.DataFrame) -> pd.Series:
        period = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int).clip(1, 10)
        sec_in_period = df["time"].apply(parse_time_to_seconds).astype(float).clip(lower=0.0)
        return (period - 1) * 1200.0 + sec_in_period

    def _compute_dt(self, df: pd.DataFrame) -> np.ndarray:
        abs_t = df["_abs_time"].to_numpy(dtype=np.float32)
        dt = np.zeros(len(df), dtype=np.float32)
        for i in range(1, len(df)):
            dt[i] = float(max(0.0, abs_t[i] - abs_t[i - 1]))
        return np.clip(dt, 0.0, 60.0).astype(np.float32)

    def _compute_score_diff(self, df: pd.DataFrame) -> np.ndarray:
        """
        ScoreDiff z pohledu team_id na řádku.
        Pozn.: pokud máš jen window, score_diff je “lokální” (od začátku okna),
        ale pořád dává smysl jako kontext.
        """
        teams = df["team_id"].astype(str).to_numpy()
        event_type = df["event_type"].astype(str).to_numpy()

        if "goal" in df.columns:
            goal_flag = pd.to_numeric(df["goal"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int32)
        else:
            goal_flag = np.array([1 if ("GOAL" in str(e).upper()) else 0 for e in event_type], dtype=np.int32)

        goals_by_team: Dict[str, int] = {}
        score_diff = np.zeros(len(df), dtype=np.float32)

        for i in range(len(df)):
            t = teams[i]
            if goal_flag[i] == 1:
                goals_by_team[t] = goals_by_team.get(t, 0) + 1

            my_goals = goals_by_team.get(t, 0)
            total_goals = sum(goals_by_team.values())
            opp_goals = total_goals - my_goals
            score_diff[i] = float(my_goals - opp_goals)

        return np.clip(score_diff, -10.0, 10.0).astype(np.float32)

    def _make_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        NUM (musí sedět s train):
          [x, y, xg, st, shot_like, dist, dt, score_diff, corsi, fenwick, shot_col, abs_time]
        EVT:
          event_id
        """
        df = standardize_columns(df)

        # sort by period+time within the provided history
        df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
        df["_t_sort"] = df["time"].apply(parse_time_to_seconds).astype(float)
        df = df.sort_values(["period", "_t_sort"], ascending=[True, True]).reset_index(drop=True)

        # abs time + dt
        df["_abs_time"] = self._compute_abs_time(df)
        dt = self._compute_dt(df)
        score_diff = self._compute_score_diff(df)
        abs_time = df["_abs_time"].to_numpy(dtype=np.float32)

        x = pd.to_numeric(df["coords_x"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        y = pd.to_numeric(df["coords_y"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        xg = pd.to_numeric(df["xg"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

        st = df["StrengthState"].apply(parse_strength).astype(np.float32).to_numpy()

        shot_like = df["event_type"].astype(str).str.contains("SHOT|GOAL|MISS|BLOCK", case=False, na=False) \
            .astype(np.float32).to_numpy()

        d1 = np.sqrt((x - 89.0) ** 2 + (y - 0.0) ** 2).astype(np.float32)
        d2 = np.sqrt((x + 89.0) ** 2 + (y - 0.0) ** 2).astype(np.float32)
        dist = np.minimum(d1, d2).astype(np.float32)

        # optional
        corsi = (pd.to_numeric(df["Corsi"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
                 if "Corsi" in df.columns else np.zeros(len(df), dtype=np.float32))
        fenwick = (pd.to_numeric(df["Fenwick"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
                   if "Fenwick" in df.columns else np.zeros(len(df), dtype=np.float32))
        shot_col = (pd.to_numeric(df["Shot"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
                    if "Shot" in df.columns else np.zeros(len(df), dtype=np.float32))

        X_num = np.stack(
            [x, y, xg, st, shot_like, dist, dt, score_diff, corsi, fenwick, shot_col, abs_time],
            axis=1
        ).astype(np.float32)

        X_evt = df["event_type"].apply(self._map_event).to_numpy(dtype=np.int64)

        # ✅ dim check against meta mean/std
        if self.numeric_mean is not None and self.numeric_std is not None:
            if X_num.shape[1] != self.numeric_mean.shape[0]:
                raise ValueError(
                    f"Numeric dim mismatch: X={X_num.shape[1]} vs mean={self.numeric_mean.shape[0]}"
                )
            X_num = (X_num - self.numeric_mean) / (self.numeric_std + 1e-6)

        return X_num, X_evt

    def _make_seq(self, history_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vytvoří (1, L, D) a (1, L) pro LSTM.
        Když je historie kratší než seq_len, doplní se nulami vlevo.
        """
        if history_df is None or len(history_df) == 0:
            raise ValueError("history_df je prázdné.")

        X_num, X_evt = self._make_features(history_df)

        L = int(self.seq_len)
        D = int(X_num.shape[1])

        # take last L rows; if shorter, left-pad zeros
        if len(X_num) >= L:
            xn = X_num[-L:]
            xe = X_evt[-L:]
        else:
            pad = L - len(X_num)
            xn = np.vstack([np.zeros((pad, D), dtype=np.float32), X_num])
            xe = np.concatenate([np.zeros((pad,), dtype=np.int64), X_evt], axis=0)

        xn_t = torch.tensor(xn[None, :, :], dtype=torch.float32, device=self.device)  # (1,L,D)
        xe_t = torch.tensor(xe[None, :], dtype=torch.long, device=self.device)        # (1,L)
        return xn_t, xe_t

    # =========================
    # PUBLIC API
    # =========================
    def predict_risk(self, history_df: pd.DataFrame) -> float:
        """
        Vrátí pravděpodobnost (0..1) pro "goal against v následujících N eventech".
        """
        if self.model is None:
            raise RuntimeError("Model není načtený.")

        xn, xe = self._make_seq(history_df)

        with torch.no_grad():
            logits = self.model(xn, xe)          # (1,)
            prob = torch.sigmoid(logits).item()

        return float(prob)

    def predict(self, history_df: pd.DataFrame) -> dict:
        """
        Vrátí {prob, is_risky,threshold}.
        """
        prob = self.predict_risk(history_df)
        return {
            "prob": float(prob),
            "threshold": float(self.threshold),
            "is_risky": bool(prob >= self.threshold),
        }
