import os
import random

# Pokusíme se importovat novou knihovnu
try:
    from google import genai
    from google.genai import types

    _HAS_NEW_SDK = True
except ImportError:
    _HAS_NEW_SDK = False


class GeminiAgent:
    """
    Moderní Gemini Agent využívající výhradně nové SDK 'google-genai'.
    Nastaven na model: gemini-2.0-flash
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        self.is_active = False
        self.last_error = None

        self.offline_rules = [
            "TLAČ DO BRÁNY! (Offline)",
            "ZJEDNODUŠIT HRU! (Offline)",
            "STŘÍLEJ BEZ PŘÍPRAVY! (Offline)",
            "POZOR NA BREJKY! (Offline)"
        ]

        # Inicializace
        if not _HAS_NEW_SDK:
            self.last_error = "Chybí knihovna 'google-genai'. Spusť: pip install google-genai"
            print(f"❌ {self.last_error}")
        elif self.api_key:
            self._init_client()
        else:
            self.last_error = "Čekám na API klíč..."

    def _init_client(self):
        try:
            # Inicializace klienta (nové SDK)
            self.client = genai.Client(api_key=self.api_key)
            self.is_active = True
            print(f"✅ GeminiAgent: Připojeno (Model: {self.model_name})")
        except Exception as e:
            self.last_error = str(e)
            self.is_active = False
            print(f"❌ GeminiAgent: Init Failed: {e}")

    def get_advice(self, context: dict) -> str:
        # Pokud není aktivní, vracíme offline radu
        if not self.is_active or not self.client:
            return self._get_offline_advice(context)

        prompt = self._build_prompt(context)

        try:
            # Volání API (nová syntaxe)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=60
                )
            )

            if response.text:
                return response.text.strip()
            return "..."

        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            self.last_error = str(e)
            # Fallback na offline
            return self._get_offline_advice(context)

    def _get_offline_advice(self, context: dict) -> str:
        return random.choice(self.offline_rules)

# agents/gemini_agent.py - stačí nahradit metodu _build_prompt

    def _build_prompt(self, context: dict) -> str:
        # Získáme syrové riziko (0.0 až 1.0)
        try:
            risk = float(context.get("ai_prob_raw", 0.0))
        except:
            risk = 0.5

        pressure = float(context.get("my_pressure_score", 0.0))

        # DYNAMICKÁ PERSONA PODLE RIZIKA
        if risk > 0.50:
            # KRIZOVÝ MÓD (Obrana)
            role = "Jsi hysterický trenér, který se bojí gólu. Tvoje obrana hoří."
            task = "KŘIČ na obránce! Přikaž jim vyhodit puk, zblokovat střelu, nebo faulovat. Používej VYKŘIČNÍKY! Buď ve stresu. Nebuď vulgární"
        elif risk < 0.35 and pressure > 1.0:
            # ÚTOČNÝ MÓD (Tlak)
            role = "Jsi agresivní útočný trenér. Cítíš krev."
            task = "Hecuj hráče do zakončení! Chceš gól hned teď. Buď stručný a dravý. Nebuď vulgární."
        else:
            # NEUTRÁLNÍ MÓD
            role = "Jsi zkušený trenér NHL."
            task = "Dej krátký taktický pokyn k aktuální hře. Žádná omáčka. Nebuď vulgární"

        return f"""
Role: {role}
Situace: {context.get("my_team")} vs {context.get("opponent_team")}.
Stav hry: Tlak {pressure:.1f}, Riziko gólu proti nám: {risk*100:.0f}%.
ÚKOL: {task}
Jazyk: Čeština (hokejový slang).
Délka: Max 1 věta.
"""