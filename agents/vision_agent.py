# agents/vision_agent.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import io


class VisionAgent:
    def __init__(self):
        pass

    def _draw_rink(self, ax):
        rink = patches.Rectangle((-100, -42.5), 200, 85, linewidth=2, edgecolor='#1e293b', facecolor='#e2e8f0',
                                 zorder=0)
        ax.add_patch(rink)
        ax.axvline(0, color='#ef4444', linewidth=2, alpha=0.3)
        ax.axvline(-25, color='#3b82f6', linewidth=2, alpha=0.3)
        ax.axvline(25, color='#3b82f6', linewidth=2, alpha=0.3)
        ax.axvline(-89, color='#ef4444', linewidth=1, alpha=0.5)
        ax.axvline(89, color='#ef4444', linewidth=1, alpha=0.5)
        for x, y in [(-69, 22), (-69, -22), (69, 22), (69, -22), (0, 0)]:
            circ = patches.Circle((x, y), 15, linewidth=1, edgecolor='#3b82f6', facecolor='none', alpha=0.2)
            ax.add_patch(circ)
            dot = patches.Circle((x, y), 0.5, color='#3b82f6', alpha=0.5)
            ax.add_patch(dot)
        ax.add_patch(patches.Arc((-89, 0), 6, 4, theta1=270, theta2=90, color='#ef4444', alpha=0.5))
        ax.add_patch(patches.Arc((89, 0), 6, 4, theta1=90, theta2=270, color='#ef4444', alpha=0.5))

    def generate_dashboard(self, df_window: pd.DataFrame, momentum_history: list | None = None,
                           team_name: str = "Opponent"):
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('#0e1117')
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

        # --- A. MAPA ---
        ax_map = fig.add_subplot(gs[0, 0])
        ax_map.set_facecolor('#0e1117')
        self._draw_rink(ax_map)

        if not df_window.empty:
            df_window["coords_x"] = pd.to_numeric(df_window["coords_x"], errors='coerce')
            df_window["coords_y"] = pd.to_numeric(df_window["coords_y"], errors='coerce')

            # 1. Střely (Shots) - Pouze 'SHOT' (bez GOAL)
            # Filtrujeme tak, aby to nebyl GÓL
            is_goal_mask = pd.Series([False] * len(df_window), index=df_window.index)
            if "goal" in df_window.columns:
                is_goal_mask = pd.to_numeric(df_window["goal"], errors='coerce').fillna(0) == 1

            # Logika pro střely: Je to SHOT/MISS/BLOCK a NENÍ to gól
            shot_types = df_window['event_type'].astype(str).str.contains('SHOT|MISS|BLOCK', case=False, na=False)
            shots = df_window[shot_types & (~is_goal_mask)]

            ax_map.scatter(shots["coords_x"], shots["coords_y"], c='#38bdf8', marker='x', s=80, alpha=0.7,
                           label='Střely', zorder=2)

            # 2. Góly (Goals) - PŘÍSNÁ LOGIKA
            # Pokud existuje sloupec 'goal' (1/0), věříme jen jemu.
            if "goal" in df_window.columns:
                raw_goals = df_window[df_window["goal"] == 1]
            else:
                # Fallback: Musí to být PŘESNĚ "GOAL", ne "SHOT ON GOAL"
                raw_goals = df_window[df_window['event_type'].astype(str).str.upper() == 'GOAL']

            # Odstranění duplicit (asistencí)
            if not raw_goals.empty:
                cols = ['period', 'time'] if 'time' in raw_goals.columns else ['period']
                goals = raw_goals.drop_duplicates(subset=cols)

                ax_map.scatter(
                    goals["coords_x"], goals["coords_y"],
                    c='#facc15', marker='*', s=250, edgecolors='black', linewidth=1.5,
                    label='Gól', zorder=3
                )

        ax_map.set_xlim(-100, 100)
        ax_map.set_ylim(-42.5, 42.5)
        ax_map.set_title("Herní situace (poslední akce)", color='white', fontsize=10)
        ax_map.axis('off')

        # --- B. MOMENTUM ---
        ax_mom = fig.add_subplot(gs[1, 0])
        ax_mom.set_facecolor('#0e1117')
        if momentum_history and len(momentum_history) > 0:
            vals = [x['value'] for x in momentum_history][-50:]
            x_vals = range(len(vals))
            colors = ['#22c55e' if v > 0 else '#ef4444' for v in vals]
            ax_mom.bar(x_vals, vals, color=colors, alpha=0.8, width=1.0)
            ax_mom.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax_mom.axis('off')

        return fig, "OK", None