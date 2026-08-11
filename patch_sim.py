import re
import os

with open('src/simulation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace simuliere_verlaeufe def
content = content.replace(
    'def simuliere_verlaeufe(studierende: List[Student], stammdaten: Dict[str, pd.DataFrame], rng: np.random.Generator):',
    'def simuliere_verlaeufe(studierende: List[Student], stammdaten: Dict[str, pd.DataFrame], block_support: bool = False):'
)

# Insert seeded RNG
content = content.replace(
    'if (idx + 1) % CONFIG["log_every_n_studis"] == 0: print(f"Simuliert: {idx+1}/{len(studierende)}")',
    'if (idx + 1) % CONFIG["log_every_n_studis"] == 0: print(f"Simuliert: {idx+1}/{len(studierende)}")\n        rng = np.random.default_rng(int(hash(studi.studierenden_id) % (2**32)))'
)

part1, part2 = content.split('            # --- Reaktive Support-Nutzung simulieren ---')
part2_a, part2_b = part2.split('            # --- Lernen und Prüfungen ---')

lines = part2_a.split('\n')
new_lines = []
for line in lines:
    if 'teilgenommene_angebote =' in line or 'support_zeit_kosten =' in line or not line.strip():
        new_lines.append(line)
    else:
        new_lines.append('    ' + line)

final_part2_a = '\n'.join(new_lines)
final_part2_a = final_part2_a.replace('        for _, angebot in support_df.iterrows():', '        if not block_support:\n            for _, angebot in support_df.iterrows():')

new_content = part1 + '            # --- Reaktive Support-Nutzung simulieren ---' + final_part2_a + '            # --- Lernen und Prüfungen ---' + part2_b

main_block = """if __name__ == "__main__":
    print("Starte True Counterfactual Trajectory Simulator (Simulator v2) ...")
    import os
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    global_rng = np.random.default_rng(42)
    stammdaten = generiere_stammdaten(global_rng)
    
    import copy
    print("Generiere Basis-Population...")
    studierende_basis = generiere_studierende(stammdaten, global_rng)
    
    studierende_A = copy.deepcopy(studierende_basis)
    studierende_B = copy.deepcopy(studierende_basis)
    
    print("\\n--- Simuliere Universum A (Support erlaubt) ---")
    simuliere_verlaeufe(studierende_A, stammdaten, block_support=False)
    
    print("\\n--- Simuliere Universum B (Support blockiert) ---")
    simuliere_verlaeufe(studierende_B, stammdaten, block_support=True)
    
    dropouts_A = sum(1 for s in studierende_A if s.abgebrochen)
    dropouts_B = sum(1 for s in studierende_B if s.abgebrochen)
    
    n = len(studierende_basis)
    rate_A = dropouts_A / n
    rate_B = dropouts_B / n
    true_rr = rate_A / rate_B if rate_B > 0 else 1.0
    
    print("\\n==========================================================================")
    print("   TRUE CAUSAL MACRO EFFECT (TRAJECTORY CLONES)")
    print("==========================================================================")
    print(f"Dropout-Rate mit Support (Klon A) : {rate_A:.2%} ({dropouts_A}/{n})")
    print(f"Dropout-Rate ohne Support (Klon B): {rate_B:.2%} ({dropouts_B}/{n})")
    print(f"Wahre Relative Risiko-Senkung (RR): {true_rr:.4f} ({(1-true_rr):.2%} Reduktion)")
    print("==========================================================================")
    
    import json
    from pathlib import Path
    out_file = Path("output_dl/metrics/true_macro_causal_effect.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"dropout_rate_with_support": rate_A, "dropout_rate_without_support": rate_B, "true_relative_risk": true_rr}, f, indent=4)
"""

new_content = re.sub(r'if __name__ == "__main__":.*', main_block, new_content, flags=re.DOTALL)

with open('src/simulation_v2.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
