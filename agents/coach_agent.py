# agents/coach_agent.py
import numpy as np
import pandas as pd

from agents.analyst_agent import AnalystAgent
from agents.vision_agent import VisionAgent
from agents.gemini_agent import GeminiAgent
from agents.predictor_agent_dl import PredictorAgentDL


class CoachAgent:
    def __init__(self, api_key: str | None):
        self.analyst = AnalystAgent()
        self.vision = VisionAgent()
        self.gemini = GeminiAgent(api_key)
        self.predictor = PredictorAgentDL()

    def set_api_key(self, api_key: str | None) -> None:
        api_key = api_key if api_key and str(api_key).strip() else None
        self.gemini = GeminiAgent(api_key)

    def analyze_game_segment(
            self,
            df_window: pd.DataFrame,
            my_team_id: str,
            opponent_id: str,
            momentum_history: list | None = None,
            use_gemini: bool = False,
    ):
        if momentum_history is None:
            momentum_history = []

        my_team_id = str(my_team_id)
        opponent_id = str(opponent_id)

        # 1) Momentum (z pohledu mého týmu)
        stats = self.analyst.analyze_momentum(df_window, my_team_id=my_team_id)
        pressure = float(stats.get("momentum_score", 0.0))

        # 2) Výpočet rizika (Context-Aware)

        # A. Hrubá síla AI modelu (detekuje anomálie / chyby v obraně aktivního týmu)
        pred = self.predictor.predict(df_window)
        dl_prob = float(pred["prob"])  # Toto je "Goal Against Active Team"

        # B. xG soupeře v okně (přímé nebezpečí střelby)
        opp_mask = df_window["team_id"].astype(str) == opponent_id
        opp_xg_sum = df_window.loc[opp_mask, "xg"].sum()

        # --- LOGIKA PERSPEKTIVY ---
        # Pokud nás soupeř drtí (záporný tlak) nebo má vysoké xG,
        # riziko musí být vysoké, i když DL model (trénovaný na chyby) mlčí.

        final_risk = 0.0

        if pressure < -1.0:
            # Soupeř tlačí -> Riziko je kombinace jeho xG a AI modelu
            # (Používáme tanh pro normalizaci xG do 0..1)
            xg_risk = np.tanh(opp_xg_sum * 1.5)
            final_risk = max(xg_risk, dl_prob)

            # Pokud soupeř extrémně tlačí, riziko neklesne pod 0.2
            final_risk = max(final_risk, 0.2)

        elif pressure > 1.0:
            # My tlačíme -> Riziko, že dostaneme gól, je nízké
            # Použijeme DL model jen kdyby detekoval fatální chybu (turnover)
            final_risk = dl_prob * 0.5  # Tlumíme riziko, protože máme puk

        else:
            # Hra ve středu pole
            final_risk = dl_prob

        # Oříznutí 0..1
        final_risk = float(np.clip(final_risk, 0.0, 0.99))

        thr = float(pred.get("threshold", 0.5))
        is_risky = bool(final_risk > 0.35)  # Custom threshold pro UI
        ai_msg = f"{final_risk * 100:.1f}%"

        # 3) Vision status
        vision_status = "SKIPPED"
        heatmap_fig = None

        if (pressure < -0.5) or is_risky or (final_risk > 0.35):
            try:
                heatmap_fig, status, _ = self.vision.generate_dashboard(
                    df_window,
                    momentum_history=momentum_history,
                    team_name=opponent_id,
                )
                vision_status = f"ACTIVE ({status})"
            except Exception:
                vision_status = "ACTIVE (ERROR)"

        # 4) Gemini
        advice = "💡 Klikni na tlačítko 'Zeptat se Coache' pro AI radu."

        if use_gemini:
            context = {
                "my_pressure_score": pressure,
                "momentum_trend": stats.get("momentum_trend", ""),
                "vision_status": vision_status,
                "ai_goal_prob": ai_msg,
                "ai_prob_raw": final_risk,
                "ai_is_risky": is_risky,
                "my_team": my_team_id,
                "opponent_team": opponent_id,
            }
            try:
                advice = self.gemini.get_advice(context)
            except Exception as e:
                advice = f"CHYBA: {str(e)}"

        return {
            "pressure_index": pressure,
            "advice": advice,
            "stats": stats,
            "heatmap": heatmap_fig,
            "ai_prob": ai_msg,
            "risk_value": final_risk,
            "threshold": thr,
            "is_risky": is_risky,
            "vision_status": vision_status,
        }