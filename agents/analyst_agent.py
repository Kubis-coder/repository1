import pandas as pd


class AnalystAgent:
    """
    Deterministický Ice Tilt / Momentum.
    Z pohledu my_team_id:
      + = my team tlačí
      - = soupeř tlačí
    """

    def __init__(self):
        self.weights = {
            "GOAL": 5.0,
            "SHOT": 1.0,
            "MISS": 0.5,
            "BLOCK": 0.5,
            "HIT": 0.3,
            "TAKEAWAY": 0.5,
            "GIVEAWAY": -0.5,
            "PENALTY": -2.0,
            "FACEOFF": 0.1,
        }
        self.decay_factor = 0.95

    def analyze_momentum(self, df_window: pd.DataFrame, my_team_id: str):
        my_team_id = str(my_team_id)
        current_momentum = 0.0

        for _, row in df_window.iterrows():
            event = str(row.get("event_type", "UNKNOWN")).upper()
            team = str(row.get("team_id", "UNKNOWN"))
            xg = float(row.get("xg", 0.0) or 0.0)

            val = 0.0
            for key, weight in self.weights.items():
                if key in event:
                    val = weight
                    break

            if val > 0:
                val += (xg * 2.0)

            # sign by coach team
            if team != my_team_id:
                val = -val

            current_momentum = (current_momentum * self.decay_factor) + val

        trend = "STABLE"
        if abs(current_momentum) > 3.0:
            trend = "HIGH PRESSURE"
        if abs(current_momentum) > 6.0:
            trend = "DOMINATION"

        return {"momentum_score": current_momentum, "momentum_trend": trend}
