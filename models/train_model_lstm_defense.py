import os
import sys
import time
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.features import standardize_columns, parse_strength, parse_time_to_seconds


# ===================== CONFIG =====================

@dataclass(frozen=True)
class Config:
    data_path: str = "data/NHL.csv"
    model_path: str = "models/nhl_brain_lstm_defense.pt"
    meta_path: str = "models/nhl_brain_lstm_defense_meta.json"

    random_seed: int = 42

    # HOKEJ: kratší horizont = méně noise, víc signálu
    # Predict: "goal AGAINST current team occurs within next N events (same game)"
    target_window: int = 7
    seq_len: int = 20               # historie

    batch_size: int = 1024
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Threshold tuning
    thresholds: Tuple[float, ...] = tuple(np.arange(0.10, 0.91, 0.05).tolist())
    min_recall: float = 0.50

    # LSTM arch
    emb_dim: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3

    # split by games
    train_frac_games: float = 0.85


CFG = Config()

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

EVENT_VOCAB = [
    "SHOT", "GOAL", "MISS", "BLOCK", "HIT", "TAKEAWAY",
    "GIVEAWAY", "PENALTY", "FACEOFF", "OTHER"
]
EVENT2ID = {e: i for i, e in enumerate(EVENT_VOCAB)}


# ===================== SEED =====================

def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== FEATURES =====================

def map_event(evt) -> int:
    s = str(evt).upper()
    for k in EVENT2ID:
        if k != "OTHER" and k in s:
            return EVENT2ID[k]
    return EVENT2ID["OTHER"]


def _get_optional_numeric(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=np.float32)
    return np.full((len(df),), float(default), dtype=np.float32)


def compute_abs_time(df: pd.DataFrame) -> pd.Series:
    """
    Robustní absolutní čas v rámci zápasu v sekundách:
    abs_sec = (period-1)*1200 + sec_in_period
    """
    period = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int).clip(1, 10)
    sec_in_period = df["time"].apply(parse_time_to_seconds).astype(float).clip(lower=0.0)
    return (period - 1) * 1200.0 + sec_in_period


def compute_dt_per_game(df: pd.DataFrame) -> np.ndarray:
    """
    dt = rozdíl času mezi aktuální a předchozí událostí v rámci stejného game_id.
    (clamp na rozumný rozsah)
    """
    abs_t = df["_abs_time"].to_numpy(dtype=np.float32)
    game = df["game_id"].astype(str).to_numpy()

    dt = np.zeros(len(df), dtype=np.float32)
    prev_t = 0.0
    prev_g = None
    for i in range(len(df)):
        g = game[i]
        t = abs_t[i]
        if prev_g is None or g != prev_g:
            dt[i] = 0.0
        else:
            dt[i] = float(max(0.0, t - prev_t))
        prev_t = t
        prev_g = g

    # hokej: dt bývá typicky 0–20s, občas větší (přerušení)
    dt = np.clip(dt, 0.0, 60.0)
    return dt


def compute_score_diff_per_game(df: pd.DataFrame) -> np.ndarray:
    """
    ScoreDiff z pohledu "team_id na řádku":
      score_diff[i] = goals(team_id[i]) - goals(opponent) do času i (v rámci game_id)
    Když Goal sloupec není, bereme event_type obsahující "GOAL".
    """
    game_ids = df["game_id"].astype(str).to_numpy()
    teams = df["team_id"].astype(str).to_numpy()
    event_type = df["event_type"].astype(str).to_numpy()

    if "goal" in df.columns:
        goal_flag = pd.to_numeric(df["goal"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        goal_flag = np.array([1 if ("GOAL" in str(e).upper()) else 0 for e in event_type], dtype=np.int32)

    score_diff = np.zeros(len(df), dtype=np.float32)

    # průběžně per game
    cur_game = None
    goals_by_team: Dict[str, int] = {}

    for i in range(len(df)):
        g = game_ids[i]
        t = teams[i]

        if cur_game is None or g != cur_game:
            cur_game = g
            goals_by_team = {}

        # update score on GOAL rows
        if goal_flag[i] == 1:
            goals_by_team[t] = goals_by_team.get(t, 0) + 1

        # score diff from POV of team on this row
        my_goals = goals_by_team.get(t, 0)
        # soupeř = všechny ostatní týmy v zápase (většinou 1 další); robustně suma - my
        total_goals = sum(goals_by_team.values())
        opp_goals = total_goals - my_goals
        score_diff[i] = float(my_goals - opp_goals)

    # clamp (reálně -6..+6 většinou bohatě)
    score_diff = np.clip(score_diff, -10.0, 10.0).astype(np.float32)
    return score_diff


def make_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hokejovější featury:
    NUM: [x, y, xg, st, shot_like, dist, dt, score_diff, corsi, fenwick, shot_col, abs_time]
    EVT: event_id (embedding)
    """
    df = standardize_columns(df)

    # absolutní čas a dt potřebujeme už tady
    df["_abs_time"] = compute_abs_time(df)
    df["game_id"] = df["game_id"].astype(str)

    x = pd.to_numeric(df["coords_x"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    y = pd.to_numeric(df["coords_y"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    xg = pd.to_numeric(df["xg"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    st = df["StrengthState"].apply(parse_strength).astype(np.float32).to_numpy()

    shot_like = df["event_type"].str.contains("SHOT|GOAL|MISS|BLOCK", case=False, na=False) \
        .astype(np.float32).to_numpy()

    # dist to nearest goal (+89,0) or (-89,0)
    d1 = np.sqrt((x - 89.0) ** 2 + (y - 0.0) ** 2).astype(np.float32)
    d2 = np.sqrt((x + 89.0) ** 2 + (y - 0.0) ** 2).astype(np.float32)
    dist = np.minimum(d1, d2)

    # ✅ new hockey signals
    dt = compute_dt_per_game(df)
    score_diff = compute_score_diff_per_game(df)
    abs_time = df["_abs_time"].to_numpy(dtype=np.float32)

    # optional (pokud existují)
    corsi = _get_optional_numeric(df, "Corsi", 0.0)
    fenwick = _get_optional_numeric(df, "Fenwick", 0.0)
    shot_col = _get_optional_numeric(df, "Shot", 0.0)

    X_num = np.stack(
        [x, y, xg, st, shot_like, dist, dt, score_diff, corsi, fenwick, shot_col, abs_time],
        axis=1
    ).astype(np.float32)

    X_evt = df["event_type"].apply(map_event).to_numpy(dtype=np.int64)
    return X_num, X_evt


# ===================== TARGET: GOAL AGAINST (FORWARD) =====================

def build_target_goal_against_forward(df: pd.DataFrame, window: int) -> np.ndarray:
    """
    y[i] = 1 pokud v příštích (i+1 .. i+window) událostech (VE STEJNÉM ZÁPASE)
           padne gól a střelec (team_id u GOAL eventu) != current team_id (v řádku i).
    """
    if "game_id" not in df.columns or "team_id" not in df.columns:
        raise ValueError("Potřebuji sloupce: game_id, team_id (po standardize_columns).")

    df = df.copy()
    df["team_id"] = df["team_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)

    if "goal" in df.columns:
        df["goal"] = pd.to_numeric(df["goal"], errors="coerce").fillna(0).astype(int)
        is_goal_all = df["goal"].to_numpy(dtype=np.int32)
    else:
        is_goal_all = df["event_type"].astype(str).str.contains("GOAL", case=False, na=False).astype_toggle = None
        is_goal_all = df["event_type"].astype(str).str.contains("GOAL", case=False, na=False).astype(int).to_numpy(dtype=np.int32)

    y = np.zeros(len(df), dtype=np.float32)

    for _, sub in df.groupby("game_id", sort=False):
        idxs = sub.index.to_numpy(dtype=np.int64)
        teams = sub["team_id"].to_numpy(dtype=object)

        # goal mask within this sub (avoid re-index issues)
        if "goal" in sub.columns:
            is_goal = sub["goal"].to_numpy(dtype=np.int32)
        else:
            is_goal = sub["event_type"].astype(str).str.contains("GOAL", case=False, na=False).astype(int).to_numpy(dtype=np.int32)

        goal_team = teams  # at GOAL rows, team_id is scoring team

        m = len(sub)
        for j in range(m):
            cur_team = teams[j]
            end = min(m, j + window + 1)
            if j + 1 >= end:
                continue

            future_goal_mask = is_goal[j + 1:end] == 1
            if not future_goal_mask.any():
                continue

            future_goal_teams = goal_team[j + 1:end][future_goal_mask]
            if np.any(future_goal_teams != cur_team):
                y[idxs[j]] = 1.0

    return y


# ===================== SPLIT BY GAME (SHUFFLED) =====================

def split_games(df: pd.DataFrame, train_frac_games: float, seed: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if "game_id" not in df.columns:
        raise ValueError("Chybí game_id pro split-by-game.")

    games = df["game_id"].astype(str).to_numpy()
    unique_games = pd.unique(games)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)

    split_g = int(len(unique_games) * train_frac_games)
    train_games = unique_games[:split_g]
    val_games = unique_games[split_g:]

    train_mask = np.isin(games, train_games)
    val_mask = ~train_mask
    return train_mask, val_mask, list(train_games), list(val_games)


# ===================== DATASET: NO CROSS-GAME SEQUENCES =====================

class HockeySeqDataset(Dataset):
    """
    Samples indices i where:
      - i has at least seq_len history (i-seq_len .. i-1)
      - all those rows are within same game_id
      - i belongs to allowed_mask (train or val)
    """
    def __init__(self, df: pd.DataFrame, X_num: np.ndarray, X_evt: np.ndarray, y: np.ndarray,
                 seq_len: int, allowed_mask: np.ndarray):
        self.df = df
        self.X_num = X_num
        self.X_evt = X_evt
        self.y = y
        self.seq_len = seq_len

        game_ids = df["game_id"].astype(str).to_numpy()

        valid_idxs: List[int] = []
        for i in range(seq_len, len(df)):
            if not allowed_mask[i]:
                continue
            g = game_ids[i]
            if np.all(game_ids[i - seq_len:i] == g):
                valid_idxs.append(i)

        self.idxs = np.array(valid_idxs, dtype=np.int64)

    def __len__(self):
        return int(len(self.idxs))

    def __getitem__(self, k: int):
        i = int(self.idxs[k])
        x_num = self.X_num[i - self.seq_len:i]
        x_evt = self.X_evt[i - self.seq_len:i]
        target = float(self.y[i])
        return (
            torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(x_evt, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )


# ===================== MODEL (LSTM) =====================

class NHL_LSTM(nn.Module):
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
        e = self.emb(x_evt)
        x = torch.cat([x_num, e], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ===================== METRICS / THRESHOLD =====================

def metrics_at_threshold(probs: np.ndarray, y_true: np.ndarray, thr: float):
    pred = probs >= thr
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum((~pred) & (y_true == 1)))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1, tp, fp, fn


def pick_threshold_prec_at_recall(probs: np.ndarray, y_true: np.ndarray,
                                  thresholds: Tuple[float, ...], min_recall: float):
    best_thr = float(thresholds[0])
    best_prec = -1.0
    best_tuple = None

    for thr in thresholds:
        prec, rec, f1, tp, fp, fn = metrics_at_threshold(probs, y_true, float(thr))
        if rec < min_recall:
            continue
        if prec > best_prec:
            best_prec = prec
            best_thr = float(thr)
            best_tuple = (prec, rec, f1, tp, fp, fn)

    if best_tuple is not None:
        return best_thr, best_prec, best_tuple, "prec_at_recall"

    best_f1 = -1.0
    for thr in thresholds:
        prec, rec, f1, tp, fp, fn = metrics_at_threshold(probs, y_true, float(thr))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_tuple = (prec, rec, f1, tp, fp, fn)

    return best_thr, best_f1, best_tuple, "f1_fallback"


# ===================== MAIN =====================

def main():
    set_all_seeds(CFG.random_seed)
    t_start = time.time()

    print("==================================================")
    print("   NHL DL TRAINING: LSTM DEFENSE (HOCKEY FEATURES + FORWARD TARGET)")
    print("==================================================")
    print(f"Device: {DEVICE}")
    print(f"SEQ_LEN={CFG.seq_len} | TARGET_WINDOW={CFG.target_window} | BATCH_SIZE={CFG.batch_size} | EPOCHS={CFG.epochs}")
    print(f"Objective: max precision with recall >= {CFG.min_recall:.2f}")
    print(f"Threshold grid: {list(CFG.thresholds)}")

    if not os.path.exists(CFG.data_path):
        print(f"❌ Soubor nenalezen: {CFG.data_path}")
        return

    df = pd.read_csv(CFG.data_path, sep=None, engine="python")
    df = standardize_columns(df)

    # ✅ robust sort inside game: period + time_sec
    if "game_id" in df.columns and "period" in df.columns and "time" in df.columns:
        df["_t_sort"] = df["time"].apply(parse_time_to_seconds).astype(float)
        df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
        df["game_id"] = df["game_id"].astype(str)
        df = df.sort_values(["game_id", "period", "_t_sort"], ascending=[True, True, True]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Features (HOCKEY)
    X_num, X_evt = make_features(df)

    # Split by games (SHUFFLE)
    train_mask, val_mask, train_games, val_games = split_games(df, CFG.train_frac_games, CFG.random_seed)
    print(f"Train games: {len(train_games)} | Val games: {len(val_games)}")

    # Target
    print("🎯 Generuji target: GOAL AGAINST (forward window, per-game)...")
    y = build_target_goal_against_forward(df, CFG.target_window)

    # Build datasets
    train_ds = HockeySeqDataset(df, X_num, X_evt, y, CFG.seq_len, train_mask)
    val_ds = HockeySeqDataset(df, X_num, X_evt, y, CFG.seq_len, val_mask)

    print(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

    # Normalize numeric using TRAIN sequences only
    train_rows_mask = np.zeros(len(df), dtype=bool)
    for i in train_ds.idxs:
        train_rows_mask[i - CFG.seq_len:i] = True

    mean = X_num[train_rows_mask].mean(axis=0)
    std = X_num[train_rows_mask].std(axis=0) + 1e-6
    X_num = (X_num - mean) / std

    # rebuild datasets with normalized X_num
    train_ds = HockeySeqDataset(df, X_num, X_evt, y, CFG.seq_len, train_mask)
    val_ds = HockeySeqDataset(df, X_num, X_evt, y, CFG.seq_len, val_mask)

    # pos_weight from effective TRAIN labels only (zůstává, ale target_window je kratší => méně “rozplizlé”)
    y_train_eff = y[train_ds.idxs]
    pos = float(np.sum(y_train_eff == 1.0))
    neg = float(np.sum(y_train_eff == 0.0))
    pos_weight = torch.tensor([neg / (pos + 1e-9)], device=DEVICE)

    print(f"Train positives (effective): {int(pos)} | negatives: {int(neg)} | pos_weight={pos_weight.item():.3f}")
    print(f"Numeric dim: {X_num.shape[1]} | Event vocab: {len(EVENT_VOCAB)}")

    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    model = NHL_LSTM(
        num_dim=X_num.shape[1],
        vocab_size=len(EVENT_VOCAB),
        emb_dim=CFG.emb_dim,
        hidden_dim=CFG.hidden_dim,
        num_layers=CFG.num_layers,
        dropout=CFG.dropout,
    ).to(DEVICE)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    os.makedirs(os.path.dirname(CFG.model_path), exist_ok=True)

    best_obj = -1.0
    best_epoch = -1
    best_thr = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_f1 = 0.0
    best_mode = ""

    for epoch in range(1, CFG.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for xn, xe, yb in train_dl:
            xn, xe, yb = xn.to(DEVICE), xe.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits = model(xn, xe)
            loss = crit(logits, yb)
            loss.backward()

            if CFG.grad_clip and CFG.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)

            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_dl))

        # Validation
        model.eval()
        probs_list, y_list = [], []
        with torch.no_grad():
            for xn, xe, yb in val_dl:
                xn, xe = xn.to(DEVICE), xe.to(DEVICE)
                logits = model(xn, xe)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                probs_list.append(probs)
                y_list.append(yb.numpy())

        probs_np = np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=np.float32)
        y_np = np.concatenate(y_list, axis=0).astype(np.int32) if y_list else np.array([], dtype=np.int32)

        base_rate = float(y_np.mean()) if len(y_np) else 0.0
        pmin = float(probs_np.min()) if len(probs_np) else 0.0
        p50 = float(np.quantile(probs_np, 0.5)) if len(probs_np) else 0.0
        p90 = float(np.quantile(probs_np, 0.9)) if len(probs_np) else 0.0
        pmax = float(probs_np.max()) if len(probs_np) else 0.0

        thr, obj, tup, mode = pick_threshold_prec_at_recall(probs_np, y_np, CFG.thresholds, CFG.min_recall)
        prec, rec, f1, tp, fp, fn = tup

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{CFG.epochs} | train_loss={avg_loss:.4f} | "
            f"val_base={base_rate:.3f} | val_prec={prec:.3f} val_rec={rec:.3f} val_f1={f1:.3f} | "
            f"thr*={thr:.2f} ({mode}={obj:.3f}) | "
            f"probs[min={pmin:.3f}, p50={p50:.3f}, p90={p90:.3f}, max={pmax:.3f}] | "
            f"{dt:.1f}s"
        )

        if obj > best_obj:
            best_obj = float(obj)
            best_epoch = epoch
            best_thr = float(thr)
            best_prec = float(prec)
            best_rec = float(rec)
            best_f1 = float(f1)
            best_mode = mode

            torch.save(model.state_dict(), CFG.model_path)

            meta: Dict[str, Any] = {
                "task": "goal_against_forward",
                "seq_len": CFG.seq_len,
                "target_window": CFG.target_window,

                "event_vocab": EVENT_VOCAB,
                "numeric_dim": int(X_num.shape[1]),
                "numeric_mean": mean.tolist(),
                "numeric_std": std.tolist(),

                "emb_dim": CFG.emb_dim,
                "hidden_dim": CFG.hidden_dim,
                "num_layers": CFG.num_layers,
                "dropout": CFG.dropout,

                "threshold": best_thr,
                "threshold_objective": "prec_at_recall",
                "min_recall": float(CFG.min_recall),

                "best_epoch": int(best_epoch),
                "best_val_base_rate": float(base_rate),
                "best_val_precision": best_prec,
                "best_val_recall": best_rec,
                "best_val_f1": best_f1,
            }
            with open(CFG.meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print("⭐ Ukládám nejlepší model + meta.")

    print("\n✅ Hotovo.")
    print(f"Best epoch: {best_epoch} | thr: {best_thr:.2f} | mode: {best_mode}")
    print(f"val_prec: {best_prec:.3f} | val_rec: {best_rec:.3f} | val_f1: {best_f1:.3f}")
    print(f"Model: {CFG.model_path}")
    print(f"Meta:  {CFG.meta_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
