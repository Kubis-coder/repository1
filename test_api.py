# debug_models.py
from google import genai
import os

# ZDE VLOŽ SVŮJ KLÍČ:
MY_KEY = "AIzaSyCZGT1saxbH9j1EUAwvT9zVzITNbYRIOLM"

print("--- ZJIŠŤUJI DOSTUPNÉ MODELY (verze 2) ---")

try:
    client = genai.Client(api_key=MY_KEY)

    print("Ptám se API...")
    for model in client.models.list():
        # Vypíšeme prostě jen název (name) a zobrazované jméno (display_name)
        # Používáme getattr pro jistotu, kdyby se zase něco změnilo
        name = getattr(model, 'name', 'Neznámé ID')
        display = getattr(model, 'display_name', '')

        print(f"👉 {name} ({display})")

except Exception as e:
    print(f"❌ CHYBA: {e}")