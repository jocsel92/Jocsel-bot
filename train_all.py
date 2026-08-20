"""
Train All Pairs — Convenience script
======================================
Trains EURUSD and GBPUSD models in sequence.

Usage (from the folder where you have your files):
    python train_all.py

Expects CSV files named:
    eurusd_5min.csv
    gbpusd_5min.csv

Models will be saved to:
    models_eurusd/xgboost_model.joblib + scaler.joblib
    models_gbpusd/xgboost_model.joblib + scaler.joblib
"""

import os
import sys

from pipeline import run_pipeline


PAIRS_AND_DATA = {
    "EURUSD": "eurusd_5min.csv",
    "GBPUSD": "gbpusd_5min.csv",
}


def main():
    for pair, default_csv in PAIRS_AND_DATA.items():
        csv_path = default_csv
        if not os.path.exists(csv_path):
            # Try common alternative names
            alternatives = [
                f"{pair.lower()}.csv",
                f"{pair}.csv",
                f"{pair.lower()}_5min.csv",
                f"data/{default_csv}",
            ]
            found = False
            for alt in alternatives:
                if os.path.exists(alt):
                    csv_path = alt
                    found = True
                    break
            if not found:
                print(f"⚠️ No se encontró CSV para {pair}. Probé: {default_csv}, {alternatives}")
                print(f"   Coloca tu archivo como '{default_csv}' y vuelve a ejecutar.")
                continue

        print(f"\n{'='*60}")
        print(f"  ENTRENANDO {pair}")
        print(f"  Datos: {csv_path}")
        print(f"  Salida: models_{pair.lower()}/")
        print(f"{'='*60}\n")

        try:
            run_pipeline(csv_path, pair=pair)
            print(f"\n✅ {pair} entrenado correctamente\n")
        except Exception as e:
            print(f"\n❌ Error entrenando {pair}: {e}\n")

    print("\n🏁 Entrenamiento completado.")
    print("Ahora puedes ejecutar: python live_bot.py")


if __name__ == "__main__":
    main()
