import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.coach_agent import CoachAgent
from utils.features import standardize_columns

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NHL AI Bench Boss",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# HIGH CONTRAST CSS
# =========================
st.markdown("""
<style>
    /* RESET & DARK MODE */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #333;
    }

    /* TEXTS */
    h1, h2, h3, p, div, span, label {
        color: #ffffff !important;
    }

    /* SCOREBOARD - MAX CONTRAST */
    .scoreboard-box {
        background: #0f0f0f;
        border: 2px solid #444;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.05);
    }
    .sc-team-col {
        text-align: center;
        width: 30%;
    }
    .sc-name {
        font-size: 24px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #aaaaaa !important;
    }
    .sc-score {
        font-size: 72px;
        font-weight: 900;
        line-height: 1;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    .sc-center {
        text-align: center;
        width: 40%;
        border-left: 2px solid #333;
        border-right: 2px solid #333;
    }
    .sc-time {
        font-size: 48px;
        font-weight: 700;
        color: #facc15 !important; /* Yellow for time */
        font-family: 'Courier New', monospace;
    }
    .sc-event {
        font-size: 18px;
        color: #38bdf8 !important; /* Blue for event */
        margin-top: 5px;
        font-weight: bold;
        text-transform: uppercase;
    }

    /* RISK CARD */
    .risk-card {
        background-color: #1a1a1a;
        border: 2px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* COACH BUBBLE */
    .coach-bubble {
        background: #111;
        border: 1px solid #444;
        border-left: 6px solid #3b82f6;
        padding: 20px;
        border-radius: 8px;
        font-size: 20px;
        font-style: italic;
        margin-top: 20px;
        color: #fff !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }

    /* TEAM SELECTOR STYLES */
    .team-selector-box {
        background: #222;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        text-align: center;
        margin-bottom: 20px;
    }
    .team-active {
        color: #22c55e !important; /* Green */
        font-weight: bold;
        border-bottom: 2px solid #22c55e;
    }
    .team-inactive {
        color: #666 !important;
    }

    /* FORM FIELDS - HIGH CONTRAST */
    input, textarea, select {
        background-color: #0b0b0b !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
        border-radius: 6px !important;
    }
    input::placeholder, textarea::placeholder {
        color: #bbbbbb !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] input {
        background-color: #0b0b0b !important;
        color: #ffffff !important;
        border-color: #555555 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #ffffff !important;
    }
    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea {
        background-color: #0b0b0b !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
    }
    .stDateInput input,
    .stTimeInput input {
        background-color: #0b0b0b !important;
        color: #ffffff !important;
        border: 2px solid #555555 !important;
    }
    .stMultiSelect, .stSelectbox {
        color: #ffffff !important;
    }
    div[role="listbox"] {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }
    div[role="option"] {
        color: #ffffff !important;
    }
    div[role="option"]:hover {
        background-color: #1f2937 !important;
    }

    /* TABLES / DATAFRAMES */
    .stDataFrame, .stTable, table {
        color: #ffffff !important;
        background-color: #0b0b0b !important;
    }
    table th, table td {
        color: #ffffff !important;
        background-color: #0b0b0b !important;
        border-color: #444444 !important;
    }

</style>
""", unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def mmss_from_seconds(sec: float) -> str:
    sec = int(max(0, round(float(sec))))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


def format_abs_time_hockey(abs_sec: int) -> str:
    abs_sec = int(max(0, abs_sec))
    period = abs_sec // 1200 + 1
    in_period = abs_sec % 1200
    return f"P{period} | {mmss_from_seconds(in_period)}"


def precalculate_momentum_for_team(game_df: pd.DataFrame, my_team_id: str) -> np.ndarray:
    weights = {"GOAL": 5.0, "SHOT": 1.0, "MISS": 0.5, "BLOCK": 0.5, "HIT": 0.3, "TAKEAWAY": 0.5, "GIVEAWAY": -0.5,
               "PENALTY": -2.0, "FACEOFF": 0.1}
    decay = 0.96
    mom = []
    cur = 0.0
    t0 = str(my_team_id)
    for _, row in game_df.iterrows():
        event = str(row.get("event_type", "")).upper()
        team = str(row.get("team_id", ""))
        xg = float(row.get("xg", 0.0) or 0.0)
        val = 0.0
        for k, w in weights.items():
            if k in event:
                val = w
                break
        if val > 0: val += (xg * 2.0)
        if team != t0: val = -val
        cur = cur * decay + val
        mom.append(cur)
    return np.array(mom, dtype=np.float32)


def plot_full_game_momentum(mom: np.ndarray, current_idx: int, game_df: pd.DataFrame):
    if mom is None or len(mom) == 0: return None

    # Downsample pro rychlost (stejné jako předtím)
    step = 1 if len(mom) < 2000 else 2
    y = mom[::step]
    x = np.arange(len(y))
    curr_plot_idx = current_idx // step

    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#111111")

    # 1. Křivka a výplň
    ax.axhline(0, color="#666", alpha=0.5, linewidth=1, linestyle="--")
    ax.plot(x, y, color="#38bdf8", linewidth=2)
    ax.fill_between(x, y, 0, where=(y >= 0), color="#38bdf8", alpha=0.3)
    ax.fill_between(x, y, 0, where=(y < 0), color="#f43f5e", alpha=0.3)

    # 2. Svislá čára (aktuální čas)
    if 0 <= curr_plot_idx < len(x):
        ax.axvline(curr_plot_idx, color="#ffffff", linewidth=2)

    # 3. ✨ VYKRESLENÍ GÓLŮ (Hvězdičky) ✨
    if game_df is not None and not game_df.empty:
        # A) Najdeme góly (Přísná logika jako ve VisionAgent)
        if "goal" in game_df.columns:
            raw_goals = game_df[game_df["goal"] == 1]
        else:
            raw_goals = game_df[game_df['event_type'].astype(str).str.upper() == 'GOAL']

        # B) Odstraníme duplicity (asistence)
        if not raw_goals.empty:
            cols = ['period', 'time'] if 'time' in raw_goals.columns else ['period']
            unique_goals = raw_goals.drop_duplicates(subset=cols)

            # C) Pro každý gól najdeme souřadnice v grafu
            for idx in unique_goals.index:
                # Přepočet indexu na souřadnice downsamplovaného grafu
                # Index v dataframe odpovídá indexu v 'mom' poli
                goal_x = idx // step

                if goal_x < len(y):
                    goal_y = y[goal_x]  # Hodnota momenta v čase gólu

                    # Vykreslení hvězdy
                    ax.scatter(
                        goal_x, goal_y,
                        c='#facc15',  # Zlatá barva
                        marker='*',  # Tvar hvězdy
                        s=180,  # Velikost
                        edgecolors='black',  # Černý obrys pro kontrast
                        linewidth=1.0,
                        zorder=10  # Ať je to úplně nahoře
                    )

    ax.set_xticks([]);
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    return fig

@st.cache_data(show_spinner=False)
def load_and_prepare(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None: return None
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except:
        return None
    df = standardize_columns(df)
    df["_sec_in_period"] = df["time"].apply(lambda x: parse_time_to_seconds(x)).astype(float)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int).clip(1, 10)
    df["_abs_game_sec"] = (df["period"] - 1) * 1200 + df["_sec_in_period"]
    df["game_id"] = df["game_id"].astype(str)
    df = df.sort_values(["game_id", "_abs_game_sec"], ascending=[True, True]).reset_index(drop=True)
    return df


def parse_time_to_seconds(val) -> float:
    if pd.isna(val): return 0.0
    s = str(val).strip()
    try:
        return float(s)
    except:
        pass
    if ":" in s:
        parts = s.split(":")
        if len(parts) >= 2:
            try:
                return float(int(parts[0]) * 60 + int(parts[1]))
            except:
                return 0.0
    return 0.0


def compute_score(game_df: pd.DataFrame, idx: int, team_a: str, team_b: str):
    sub = game_df.iloc[: idx + 1]
    if "goal" in sub.columns:
        is_goal = pd.to_numeric(sub["goal"], errors="coerce").fillna(0).astype(int) == 1
    else:
        is_goal = sub["event_type"].astype(str).str.contains("GOAL", case=False, na=False)
    goals = sub[is_goal]
    return int((goals["team_id"].astype(str) == str(team_a)).sum()), int(
        (goals["team_id"].astype(str) == str(team_b)).sum())


def _safe_reinit_gemini(coach: CoachAgent, api_key: str | None):
    if not api_key: return
    try:
        if hasattr(coach, "set_api_key"):
            coach.set_api_key(api_key)
        elif hasattr(coach, "gemini"):
            coach.gemini.api_key = api_key
            if hasattr(coach.gemini, "_init_client"): coach.gemini._init_client()
    except:
        pass


# =========================
# MAIN APP
# =========================
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Konfigurace")
        api_key = st.text_input("Gemini API Key", type="password")
        st.divider()
        uploaded_file = st.file_uploader("📂 Nahrát Zápas (CSV)", type=["csv"])

        st.divider()
        with st.expander("❓ Legenda"):
            st.info("Nahoru = Tlačíš Ty\nDolů = Tlačí Soupeř")

    df = load_and_prepare(uploaded_file)
    if df is None or df.empty:
        st.title("🏒 NHL Bench Boss")
        st.warning("👈 Začni nahráním CSV souboru vlevo.")
        return

    # --- GAME SELECTOR (Better UI) ---
    game_ids = df["game_id"].astype(str).unique().tolist()

    st.subheader("1. Vyber Zápas")
    col_sel1, col_sel2 = st.columns([4, 1])
    with col_sel1:
        selected_game = st.selectbox("ID Zápasu", game_ids, label_visibility="collapsed")
    with col_sel2:
        if st.button("🔄 Reset"): st.rerun()

    game_df = df[df["game_id"].astype(str) == str(selected_game)].reset_index(drop=True)

    # --- TEAM SELECTOR (High Contrast & Clear) ---
    teams = sorted(game_df["team_id"].astype(str).unique().tolist())
    if len(teams) < 2: teams = (teams + ["HOME", "AWAY"])[:2]

    if "my_team_idx" not in st.session_state: st.session_state["my_team_idx"] = 0

    def swap():
        st.session_state["my_team_idx"] = 1 - st.session_state["my_team_idx"]

    my_team = teams[st.session_state["my_team_idx"]]
    opp_team = teams[1 - st.session_state["my_team_idx"]]

    st.subheader("2. Koho trénuješ?")

    # Vlastní "Selector Box" pomocí sloupců
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.markdown(f"""
        <div class="team-selector-box" style="border: 2px solid #22c55e;">
            <div style="font-size:12px; color:#22c55e;">AKTIVNÍ KOUČ</div>
            <div style="font-size:28px; font-weight:bold; color:#fff;">{my_team}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.button("⇄ PROHODIT", on_click=swap, use_container_width=True, type="primary")
    with c3:
        st.markdown(f"""
        <div class="team-selector-box" style="opacity: 0.5;">
            <div style="font-size:12px; color:#aaa;">SOUPEŘ</div>
            <div style="font-size:28px; font-weight:bold; color:#fff;">{opp_team}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- AGENT INIT ---
    if "coach" not in st.session_state:
        st.session_state["coach"] = CoachAgent(api_key)
    else:
        _safe_reinit_gemini(st.session_state["coach"], api_key)
    coach = st.session_state["coach"]

    mom_key = f"mom::{selected_game}::{my_team}"
    if mom_key not in st.session_state: st.session_state[mom_key] = precalculate_momentum_for_team(game_df, my_team)
    mom_series = st.session_state[mom_key]

    # --- TIME SLIDER (Fixing the sprintf error) ---
    max_sec = int(game_df["_abs_game_sec"].max())

    st.subheader(f"3. Herní Čas")

    # NEPOUŽÍVÁME 'format=' uvnitř slideru, protože to háže chyby s textem
    chosen_abs = st.slider("Time", 0, max_sec, min(60, max_sec), label_visibility="collapsed")

    # Čas zobrazíme hezky nad/pod sliderem
    time_display = format_abs_time_hockey(chosen_abs)
    st.markdown(f"<h2 style='text-align:center; color:#facc15; margin:0;'>⏱️ {time_display}</h2>",
                unsafe_allow_html=True)

    # Get Index
    event_idx = int((game_df["_abs_game_sec"] - chosen_abs).abs().idxmin())
    last_row = game_df.iloc[event_idx]

    # Scoreboard data
    period = int(last_row.get("period", 1))
    sec = float(last_row.get("_sec_in_period", 0.0))
    time_str = f"P{period} {mmss_from_seconds(sec)}"
    evt_name = str(last_row.get("event_type", "START"))
    sc_my, sc_opp = compute_score(game_df, event_idx, my_team, opp_team)

    # --- SCOREBOARD RENDER (No indentation to avoid code block) ---
    # Používám 'textwrap.dedent' styl tím, že to nalepím vlevo
    st.markdown(f"""
<div class="scoreboard-box">
    <div class="sc-team-col">
        <div class="sc-name">{my_team}</div>
        <div class="sc-score">{sc_my}</div>
    </div>
    <div class="sc-center">
        <div class="sc-time">{time_str}</div>
        <div class="sc-event">{evt_name}</div>
    </div>
    <div class="sc-team-col">
        <div class="sc-name">{opp_team}</div>
        <div class="sc-score">{sc_opp}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2 = st.tabs(["🎥 LIVE KOUČINK", "📊 CELKOVÝ GRAF"])

    window = 12
    df_win = game_df.iloc[max(0, event_idx - window): event_idx + 1].copy()
    mom_hist = [{"value": float(v)} for v in mom_series[max(0, event_idx - 60):event_idx + 1]]

    with tab1:
        col_L, col_R = st.columns([1.5, 1])
        with col_L:
            st.markdown("##### 📍 Situace na ledě")
            fig, _, _ = coach.vision.generate_dashboard(df_win, momentum_history=mom_hist, team_name=str(opp_team))
            st.pyplot(fig, use_container_width=True)

        with col_R:
            st.markdown("##### 🧠 AI Asistent")
            ask = st.button("ZEPTAT SE AI COACHE ✨", type="primary", use_container_width=True)
            res = coach.analyze_game_segment(df_win, str(my_team), str(opp_team), mom_hist, use_gemini=ask)

            risk = float(res.get("risk_value", 0.0))

            # Barvy pro riziko (Kontrastní)
            if risk > 0.6:
                bc, tc, txt = "#7f1d1d", "#fca5a5", "KRITICKÉ RIZIKO"  # Red
            elif risk > 0.35:
                bc, tc, txt = "#713f12", "#fde047", "ZVÝŠENÉ RIZIKO"  # Yellow
            else:
                bc, tc, txt = "#14532d", "#86efac", "BEZPEČÍ / ÚTOK"  # Green

            st.markdown(f"""
            <div class="risk-card" style="background-color: {bc}; border-color: {tc};">
                <div style="color:{tc}; font-weight:bold; letter-spacing:1px;">{txt}</div>
                <div style="font-size:60px; font-weight:900; color:#fff;">{risk * 100:.0f}%</div>
                <div style="font-size:14px; color:#ddd;">Pravděpodobnost gólu soupeře</div>
            </div>
            """, unsafe_allow_html=True)
            # Rada
            advice = res.get("advice", "")
            if ask:
                st.session_state["last_adv"] = advice
            elif "last_adv" in st.session_state:
                advice = st.session_state["last_adv"] + " (Minule)"
            elif "Klikni" in advice:
                advice = "Čekám na signál..."

            st.markdown(f"<div class='coach-bubble'>{advice}</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("##### 📈 Vývoj Tlaku (Celý zápas)")
        # ✅ Přidáno game_df=game_df
        fig_full = plot_full_game_momentum(mom_series, event_idx, game_df=game_df)
        if fig_full: st.pyplot(fig_full, use_container_width=True)

if __name__ == "__main__":
    main()
