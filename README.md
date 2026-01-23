# 🏒 NHL AI Predictor (Bench Boss)

**NHL AI Predictor** je pokročilý nástroj pro hokejové trenéry a analytiky. Kombinuje klasickou statistiku (Ice Tilt/Momentum) s hlubokým učením (LSTM) a generativní umělou inteligencí (Google Gemini 2.0) pro predikci rizika a taktické poradenství v reálném čase.

## 🚀 Klíčové Funkce

* **📊 Analyst Agent:** Počítá momentum ("Ice Tilt") na základě střel, hitů a xG v reálném čase.
* **🧠 Predictor Agent (Deep Learning):** LSTM neuronová síť (PyTorch) analyzuje sekvenci posledních 20 herních událostí a předpovídá pravděpodobnost inkasovaného gólu.
* **🤖 Coach Agent (Gemini 2.0):** AI asistent, který sleduje hru a dává taktické rady (např. "Zjednoduš hru!", "Využij tlaku!"). Mění svou osobnost podle míry rizika.
* **🎥 Vision Agent:** Vizualizace herních situací a heatmap na kluzišti.
* **⏱️ Live Dashboard:** Interaktivní webová aplikace postavená na Streamlitu.

## 📂 Struktura Projektu

```text
nhl_ai_predictor/
├── agents/                 # Logika AI agentů (Coach, Gemini, Vision, Analyst)
├── data/                   # Vstupní data (např. NHL.csv)
├── models/                 # Uložené AI modely (.pt) a metadata
├── output/                 # Výstupy a logy
├── utils/                  # Pomocné funkce (zpracování features)
├── app.py                  # Hlavní spouštěcí soubor aplikace (Streamlit)
├── train_model_lstm_defense.py # Skript pro trénování LSTM sítě
└── requirements.txt        # Seznam potřebných knihoven
🛠 Instalace a Spuštění
1. Příprava prostředí

Doporučujeme vytvořit virtuální prostředí (venv), aby se nemíchaly knihovny:

Bash
# Vytvoření venv (Windows)
python -m venv .venv

# Aktivace (Windows)
.venv\Scripts\activate

# Aktivace (macOS/Linux)
source .venv/bin/activate
2. Instalace závislostí

Nainstalujte potřebné knihovny ze souboru requirements.txt:

Bash
pip install -r requirements.txt
3. Nastavení API klíče (pro AI Coache)

Pro funkčnost Gemini Agenta je potřeba Google API Key.

Vytvořte si klíč v Google AI Studio.

Klíč můžete zadat přímo v aplikaci do postranního panelu.

4. Spuštění aplikace

Aplikaci spustíte příkazem:

Bash
streamlit run app.py
Aplikace se otevře ve vašem prohlížeči na adrese http://localhost:8501