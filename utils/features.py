# utils/features.py
import pandas as pd


def parse_time_to_seconds(val) -> float:
    """
    Převede čas na sekundy.
    - Pokud je val číslo (např. 732.0), vrátí float.
    - Pokud je val "MM:SS" nebo "M:SS", převede na sekundy.
    - Jinak 0.0
    """
    if pd.isna(val):
        return 0.0

    s = str(val).strip()

    # numeric seconds
    try:
        return float(s)
    except Exception:
        pass

    # "MM:SS"
    if ":" in s:
        parts = s.split(":")
        if len(parts) >= 2:
            try:
                mm = int(parts[0])
                ss = int(parts[1])
                return float(mm * 60 + ss)
            except Exception:
                return 0.0

    return 0.0


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapuje tvoje reálné sloupce (nhl_sample_200_games.csv / NHL.csv) na interní názvy.
    """
    df = df.copy()

    col_map = {
        # event
        "Event": "event_type",
        "EVENT": "event_type",
        "event": "event_type",

        # team
        "Team": "team_id",
        "TEAM": "team_id",
        "Venue": "team_id",   # legacy

        # coords
        "x": "coords_x",
        "X": "coords_x",
        "y": "coords_y",
        "Y": "coords_y",

        # xG
        "xG_F": "xg",
        "xG": "xg",
        "xg": "xg",

        # strength
        "StrengthState": "StrengthState",
        "strength": "StrengthState",

        # goal
        "Goal": "goal",
        "goal": "goal",

        # game/time
        "GameID": "game_id",
        "game_id": "game_id",
        "gameTime": "time",   # v sample je to seconds-in-period
        "GameTime": "time",
        "Time": "time",
        "time": "time",

        "Period": "period",
        "period": "period",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # fallbacks
    if "event_type" not in df.columns:
        df["event_type"] = "UNKNOWN"
    if "team_id" not in df.columns:
        df["team_id"] = "UNKNOWN"
    if "coords_x" not in df.columns:
        df["coords_x"] = 0.0
    if "coords_y" not in df.columns:
        df["coords_y"] = 0.0
    if "xg" not in df.columns:
        df["xg"] = 0.0
    if "StrengthState" not in df.columns:
        df["StrengthState"] = ""
    if "period" not in df.columns:
        df["period"] = 1
    if "time" not in df.columns:
        df["time"] = 0
    if "game_id" not in df.columns:
        df["game_id"] = "GAME_1"

    # cleaning types
    df["event_type"] = df["event_type"].astype(str).str.upper()

    for col in ["coords_x", "coords_y", "xg"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # goal numeric if exists
    if "goal" in df.columns:
        df["goal"] = pd.to_numeric(df["goal"], errors="coerce").fillna(0).astype(int)

    # period numeric if exists
    if "period" in df.columns:
        df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)

    return df


def parse_strength(val) -> int:
    """
    +1 = výhoda (např 5v4 / 4v3), -1 = nevýhoda (4v5 / 3v4), 0 = rovnovážný stav.
    """
    s = str(val).lower().replace(" ", "")
    if "5v4" in s or "4v3" in s:
        return 1
    if "4v5" in s or "3v4" in s:
        return -1
    return 0
