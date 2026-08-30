# Systematischer Sensitivitätsbericht: V4 Simulations-Gridsearch

Dieser Bericht analysiert die Ergebnisse der **12 systematischen Simulations-Szenarien** (jeweils simuliert über alle 8 Universen A–H mit $N=25.000$ Studierenden pro Universum, insgesamt 96 Simulationen mit identischem Seed).

## 1. Synoptische Haupttabelle: Kausale Makro-Effekte

| Szenario | Dimension | Drop Uni A | Drop Uni B | RR (B vs. A) | Gesamt-Schutz | Nur Fachl. (F) | Nur Überf. (G) | Nur Psych. (H) | Synergie | First-Gen Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline V4 (Standard)** | Baseline | 27.84 % | 34.99 % | **1.2569** | **20.44 %** | 5.77 % | 8.85 % | 6.22 % | -0.40 %p | +4.61 %p |
| **Support-Wirkung Halbiert (0.5x)** | Support-Wirkung | 34.58 % | 34.99 % | **1.0120** | **1.19 %** | -1.02 % | 0.95 % | 0.07 % | +1.19 %p | +0.32 %p |
| **Support-Wirkung Verdoppelt (2.0x)** | Support-Wirkung | 31.72 % | 34.99 % | **1.1032** | **9.35 %** | 2.10 % | 4.30 % | 3.35 % | -0.40 %p | +2.15 %p |
| **Notenboost Fachlich Halbiert (0.04)** | Notenboost Fachlich | 28.74 % | 34.99 % | **1.2175** | **17.87 %** | 2.64 % | 8.85 % | 6.22 % | +0.16 %p | +3.60 %p |
| **Notenboost Fachlich Verdoppelt (0.16)** | Notenboost Fachlich | 26.07 % | 34.99 % | **1.3423** | **25.50 %** | 12.83 % | 8.85 % | 6.22 % | -2.39 %p | +4.50 %p |
| **Notenboost Fachlich Vervierfacht (0.32)** | Notenboost Fachlich | 26.33 % | 34.99 % | **1.3289** | **24.75 %** | 11.97 % | 8.85 % | 6.22 % | -2.29 %p | +3.24 %p |
| **Stochastisches Rauschen Halbiert (0.09)** | Rauschen | 25.44 % | 30.89 % | **1.2141** | **17.64 %** | 5.58 % | 8.52 % | 4.97 % | -1.44 %p | +2.96 %p |
| **Stochastisches Rauschen Verdoppelt (0.36)** | Rauschen | 31.49 % | 38.67 % | **1.2282** | **18.58 %** | 5.16 % | 8.79 % | 6.81 % | -2.18 %p | +3.22 %p |
| **Support-Kosten 0h (Kostenlos)** | Zeitkosten | 27.12 % | 34.99 % | **1.2905** | **22.51 %** | 9.39 % | 9.66 % | 6.78 % | -3.32 %p | +4.22 %p |
| **Support-Kosten 60h (Hohe Belastung)** | Zeitkosten | 28.45 % | 34.99 % | **1.2300** | **18.70 %** | 5.66 % | 9.61 % | 7.03 % | -3.60 %p | +3.04 %p |
| **RCT / Random Uptake (Kein Selektionsbias)** | Selektion | 18.02 % | 34.99 % | **1.9414** | **48.49 %** | 8.71 % | 31.42 % | 33.77 % | -25.41 %p | +1.49 %p |
| **Synergie-Optimum (Mult=2.0, Boost=0.16, Cost=15h)** | Synergie | 30.60 % | 34.99 % | **1.1437** | **12.56 %** | 5.38 % | 4.37 % | 2.38 % | +0.43 %p | +3.15 %p |

## 2. Detaillierte Dimensionen-Analyse

### A. Dimension Support-Wirkungs-Multiplikator (`support_effect_multiplier`)
- **Halbiert (0.5x):** Gesamt-Schutz sinkt von 20.44% auf 1.19%. RR(B vs A) sinkt auf 1.0120.
- **Verdoppelt (2.0x):** Gesamt-Schutz steigt von 20.44% auf 9.35%. RR(B vs A) steigt auf 1.1032.

### B. Dimension Notenboost Fachlich (`gewicht_support_boost`)
- **Halbiert (0.04):** Isolierte fachliche Schutzwirkung sinkt auf 2.64%.
- **Verdoppelt (0.16):** Isolierte fachliche Schutzwirkung steigt auf 12.83%.
- **Vervierfacht (0.32):** Isolierte fachliche Schutzwirkung erreicht 11.97%.

### C. Dimension Stochastisches Rauschen (`gewicht_rauschen`)
- **Halbiertes Rauschen (0.09):** In deterministischerer Umgebung beträgt RR(B vs A) 1.2141.
- **Verdoppeltes Rauschen (0.36):** Bei starkem Rauschen beträgt RR(B vs A) 1.2282.

### D. Dimension Support-Zeitkosten (`support_kosten_override`)
- **Kostenlos (0h):** Bei 0h Zeitaufwand steigt der Gesamtschutz auf 22.51% (keine Workload-Verdrängung).
- **Hohe Belastung (60h):** Bei 60h Zeitaufwand wurden 92127 Module abgeworfen, Schutzwirkung sinkt auf 18.70%.

### E. Dimension Selektions-Endogenität (`rct_support_uptake`)
- **RCT (Random Uptake):** Bei zufälliger Zuweisung (ohne Risikoselektion) beträgt die Schutzwirkung 48.49%.

### F. Synergie-Optimum (`S12_high_synergy`)
- **Maximaler Hebel:** Schutzwirkung 12.56%, RR(B vs A) = 1.1437, Synergie = +0.43 %p.

## 3. Visualisierung

![V4 Sensitivitätsanalyse Plot](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/plots_v4_sensitivity_grid.png)