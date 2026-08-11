"""
Master-Orchestrierung: Vollständiger Nachtlauf
===============================================
Führt in Reihenfolge aus:
1. Simulation V2 (5 Universen mit vollständigem Datenexport)
   → Universe A = Baseline → output_dl/
   → Universe B-E → output_dl/universe_{B,C,D,E}/
2. Ground-Truth Berechnung (Mikro-Effekte)
3. Alle Modell-Trainings (20+ Modelle)
4. Counterfactual-Analysen

Hinweis: simulation_v2.py erzeugt Universum A als Baseline UND exportiert
die CSVs direkt nach output_dl/. Ein separater main.py-Lauf ist nicht nötig.
"""

import os
import sys
import time
from pathlib import Path

# Projektstruktur
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add src to sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_step(name: str, func, *args, **kwargs):
    """Wrapper mit Zeitmessung und Error-Handling."""
    print(f"\n{'='*70}")
    print(f"  SCHRITT: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  [OK] {name} abgeschlossen in {elapsed/60:.1f} Min.")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FEHLER] bei {name} nach {elapsed/60:.1f} Min.: {e}")
        import traceback
        traceback.print_exc()
        # Weiter mit dem nächsten Schritt
        return False
    return True

def main():
    total_start = time.time()
    
    print("=" * 70)
    print("   MASTER-ORCHESTRIERUNG: VOLLSTÄNDIGER NACHTLAUF")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # =========================================================================
    # SCHRITT 1: Simulation V2 (erzeugt alle Daten + 5 Universen)
    # =========================================================================
    # simulation_v2.py exportiert Universe A direkt nach output_dl/
    # und Universe B-E nach output_dl/universe_{B,C,D,E}/
    def run_simulation_v2():
        import simulation_v2
        # Wird über if __name__ == "__main__" Block nicht getriggert bei import,
        # daher manuell:
        from simulation_v2 import simuliere_verlaeufe, generiere_stammdaten, generiere_studierende
        from config import CONFIG
        from export import as_dataframe, exportiere_csv
        from aggregate import aggregiere_daten
        import numpy as np
        import json
        
        base_output = Path(CONFIG["output_dir"])
        os.makedirs(base_output / "metrics", exist_ok=True)
        
        stammdaten = generiere_stammdaten()
        
        UNIVERSES = {
            "A": {"label": "Alle Support-Typen erlaubt",       "block_fach": False, "block_uebf": False, "block_psych": False},
            "B": {"label": "Kein Support (komplett blockiert)",  "block_fach": True,  "block_uebf": True,  "block_psych": True},
            "C": {"label": "Kein fachlicher Support",           "block_fach": True,  "block_uebf": False, "block_psych": False},
            "D": {"label": "Kein ueberfachlicher Support",      "block_fach": False, "block_uebf": True,  "block_psych": False},
            "E": {"label": "Kein psychosozialer Support",       "block_fach": False, "block_uebf": False, "block_psych": True},
        }
        
        results = {}
        POPULATION_SEED = 12345
        
        for uni_key, uni_cfg in UNIVERSES.items():
            print(f"\n  UNIVERSUM {uni_key}: {uni_cfg['label']}")
            rng = np.random.default_rng(POPULATION_SEED)
            studierende = generiere_studierende(stammdaten, rng)
            
            simuliere_verlaeufe(
                studierende, stammdaten,
                block_fach=uni_cfg["block_fach"],
                block_uebf=uni_cfg["block_uebf"],
                block_psych=uni_cfg["block_psych"]
            )
            
            n = len(studierende)
            dropouts = sum(1 for s in studierende if s.abgebrochen)
            rate = dropouts / n
            results[uni_key] = {"label": uni_cfg["label"], "dropout_rate": rate, "dropouts": dropouts, "n": n}
            print(f"    Dropout-Rate = {rate:.2%} ({dropouts}/{n})")
            
            uni_output = base_output if uni_key == "A" else base_output / f"universe_{uni_key}"
            os.makedirs(uni_output, exist_ok=True)
            df_dict = stammdaten.copy()
            df_dict.update(as_dataframe(studierende, stammdaten))
            exportiere_csv(df_dict, uni_output)
            aggregiere_daten(uni_output)
        
        # Makro-Effekte speichern
        rate_A = results["A"]["dropout_rate"]
        macro_effects = {"universe_A_baseline": {"dropout_rate": rate_A}}
        for uni_key in ["B", "C", "D", "E"]:
            rate_X = results[uni_key]["dropout_rate"]
            rr = rate_A / rate_X if rate_X > 0 else 1.0
            macro_effects[f"universe_{uni_key}"] = {
                "label": results[uni_key]["label"],
                "dropout_rate": rate_X,
                "vs_A_absolute_diff": rate_A - rate_X,
                "vs_A_relative_risk": rr,
                "vs_A_relative_reduction_pct": (1 - rr) * 100
            }
        
        out_file = base_output / "metrics" / "true_macro_effects_v2.json"
        with open(out_file, "w") as f:
            json.dump(macro_effects, f, indent=4)
        print(f"\n  Makro-Effekte gespeichert in: {out_file}")
    
    run_step("1. Simulation V2 (5 Universen)", run_simulation_v2)
    
    # =========================================================================
    # SCHRITT 2: Validierung
    # =========================================================================
    def run_validation():
        from validate import validiere_und_dokumentiere
        validiere_und_dokumentiere(Path("../output_dl"))
    
    run_step("2. Datenvalidierung", run_validation)
    
    # =========================================================================
    # SCHRITT 3: Ground-Truth Berechnung (Mikro-Effekte)
    # =========================================================================
    def run_ground_truth():
        from calculate_true_effect import main as calc_true
        calc_true()
    
    run_step("3. Ground-Truth Mikro-Effekte", run_ground_truth)
    
    # =========================================================================
    # SCHRITT 4: Alle Modell-Trainings
    # =========================================================================
    def run_all_models():
        from run_all_experiments import run_all
        run_all()
    
    run_step("4. Alle Modell-Trainings (20+ Modelle)", run_all_models)
    
    # =========================================================================
    # ZUSAMMENFASSUNG
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"   NACHTLAUF ABGESCHLOSSEN")
    print(f"   Ende: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Gesamtdauer: {total_elapsed/3600:.1f} Stunden ({total_elapsed/60:.0f} Min.)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
