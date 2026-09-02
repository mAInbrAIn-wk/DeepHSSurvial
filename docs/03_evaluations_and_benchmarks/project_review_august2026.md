# Projekt-Review: Abschlussprojekt (Stand August 2026)

**Reviewer:** Antigravity (Claude Opus 4.6, Thinking Mode)  
**Stand:** 10. August 2026, 23:00 Uhr  
**Gegenstand:** Systematischer Re-Review des Abschlussprojekts nach Implementierung der erweiterten Analyse-Pipeline (Simulator V2, Orakel-Modelle, DML, Hidden Variables Panel)  
**Referenz:** [Ursprüngliches Portfolio Review](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/portfolio_review.md)

---

## Vorbemerkung zur Methodik dieses Reviews

Dieses Review basiert auf einer vollständigen Auswertung aller 50 Metriken-JSON-Dateien in `output_dl/metrics/`, dem Quellcode aller 51 Python-Skripte in `src/`, sowie aller bisherigen Artefakte. Es wurde bewusst **kein cherry-picking** betrieben: Alle Modelle werden systematisch dargestellt, auch und gerade dort, wo die Ergebnisse unbequem sind.

---

## Teil I: Was hat sich seit dem Portfolio Review verbessert?

Das ursprüngliche Review identifizierte fünf zentrale Lücken. Hier der ehrliche Abgleich:

| Lücke (Portfolio Review) | Status | Bewertung |
| :--- | :---: | :--- |
| PH-Annahmen-Diagnose (Schoenfeld) | ✅ Implementiert | `extended_cox_delta.py` berechnet nun Schoenfeld-Residuen. Ergebnis: PH-Annahme vertretbar (mittlere Residuen 0.34–0.44). |
| Ground-Truth-Vergleich | ✅ Implementiert | `calculate_true_effect.py` + `simulation_v2.py` liefern Mikro- und Makro-Ground-Truth. |
| Kalibrierungskurven | ✅ Implementiert | `plot_calibration_curves.py` erstellt Reliability Diagrams für DTL (Brier 0.045). |
| Dashboard reparieren | ❌ Offen | README markiert es als "Work in Progress". Bleibt ein Schwachpunkt. |
| Sensitivitätsanalyse | ❌ Offen | Keine systematische Variation der Generierungsparameter. |

Darüber hinaus wurden folgende **neue Analysen** hinzugefügt:

- **Simulator V2** (Trajektorien-Klon): Paralleluniversum-Simulation mit deterministischem RNG
- **Orakel-Modelle**: ROC-AUC Lift durch Hidden Variables  
- **Hidden-Variables-Panel**: `build_delta_panel()` aggregiert jetzt `hidden_motivation_prev`, `hidden_soziale_integration_prev`, `hidden_erwartete_note_prev`
- **KI-Transparenz**: README enthält vollständigen AI-Stack
- **Dokumentationsüberarbeitung**: Präsentation, README, Cross-Referenz zum DataAnalysis-Projekt

> [!NOTE]
> Die drei gewichtigsten offenen Punkte aus dem Portfolio Review (PH-Diagnose, Ground-Truth, Kalibration) sind damit geschlossen. Das ist ein substanzieller Fortschritt.

---

## Teil II: Systematischer Modellvergleich — ALLE Modelle, ALLE Metriken

### A. Diskriminative Leistung (ROC-AUC) — Vollständige Rangfolge

| # | Modell | Scope | ROC-AUC | PR-AUC | Brier |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | Recurrent Exam Survival v2 (GRU) | Exam-Seq | **0.9020** | 0.2534 | 0.0182 |
| 2 | Extended Logistic Hazard (Exam) | Exam | 0.8945 | 0.1928 | 0.0190 |
| 3 | Logistic Hazard (Landmark) | Semester | 0.8985 | 0.8287 | — |
| 4 | Recurrent Exam Survival Delta | Exam-Seq | 0.8713 | 0.1804 | 0.0193 |
| 5 | Recurrent Exam Survival v1 | Exam-Seq | 0.8708 | 0.1640 | 0.0194 |
| 6 | Transformer Exam Survival | Exam-Seq | 0.8531 | 0.1335 | 0.0198 |
| 7 | Dynamic DeepHit Delta | Dropout | 0.8276 | 0.2944 | 0.0429 |
| 8 | Dynamic DeepHit Competing | Dropout | 0.8261 | 0.2847 | 0.0434 |
| 9 | Transformer Survival (Semester) | Semester-Seq | 0.8247 | 0.2926 | 0.0430 |
| 10 | GRU Survival (Blind) | Semester-Seq | 0.8235 | 0.2883 | 0.0433 |
| 11 | GRU Survival (Standard) | Semester-Seq | 0.8223 | 0.2841 | 0.0434 |
| 12 | Transformer Survival (Blind) | Semester-Seq | 0.8224 | 0.2898 | 0.0432 |
| 13 | Recurrent Survival Delta | Panel | 0.8229 | 0.2840 | 0.0433 |
| 14 | Extended Logistic Hazard (Panel) | Panel | 0.8005 | 0.2450 | 0.0448 |
| 15 | **DML Orthogonal Survival** | Panel | 0.7979 | 0.2219 | 0.0454 |
| 16 | Extended Logistic Hazard (Delta) | Panel | 0.7992 | 0.2278 | 0.0452 |
| 17 | DeepSurv (Landmark) | Semester | — | — | — |
| 18 | Extended DeepSurv (Delta) | Panel | **0.5618** | 0.0706 | — |
| 19 | Extended DeepSurv (Panel) | Panel | 0.5351 | 0.0659 | — |
| 20 | Extended DeepSurv (Exam) | Exam | **0.4571** | 0.0203 | — |

> [!WARNING]
> **Die drei DeepSurv-Varianten (Zeilen 18–20) sind faktisch gescheitert.** ROC-AUC-Werte unter 0.57 bedeuten, dass diese Modelle kaum besser als Zufall diskriminieren. Der `breslow_cox_loss` mit dem neuronalen Netzwerk konvergiert offensichtlich nicht zuverlässig. Das betrifft auch die **Orakel-Analyse**, deren DeepSurv-Baseline bei 0.52 liegt — ein "Lift" von +0.006 auf einem nicht-funktionierenden Modell ist bedeutungslos.

### B. Kontrafaktische Relative-Risk-Schätzungen — ALLE Modelle

Dies ist die kritischste Tabelle: Die Simulation hat den Support als **protektiv** programmiert (`gewicht_support_boost = 0.04`). Ein korrekt arbeitendes Modell sollte daher RR < 1.0 (bzw. HR < 1.0) für den fachlichen Support schätzen.

| Modell | Mean RR/HR **Fachlich** | Mean RR/HR **Überfachlich** | Mean RR/HR **Psychosozial** | Korrekte Richtung? |
| :--- | :---: | :---: | :---: | :---: |
| **Extended Cox Panel** (time-varying) | 0.959 | 0.980 | 0.955 | ✅✅✅ |
| **DML Orthogonal Survival** | 0.919 | 1.158 | 1.030 | ✅❌❌ |
| **Counterfactual HR Delta** (DeepSurv) | 0.919 | 1.090 | 0.928 | ✅❌✅ |
| **Counterfactual HR Analyzer** (DeepSurv) | 0.878 | 0.950 | 1.091 | ✅✅❌ |
| **CF RR Logistic Hazard Delta** | 0.939 | 1.105 | 1.028 | ✅❌❌ |
| **Extended Cox Delta** (delta-features) | 1.007 | 1.290 | 1.009 | ❌❌❌ |
| **CF DeepHit Fixed** | 1.050 | 1.053 | 1.046 | ❌❌❌ |
| **CF RNN Delta** | 1.077 | 1.052 | 1.494 | ❌❌❌ |
| **CF RNN Semester Delta** | 1.340 | 1.820 | 1.470 | ❌❌❌ |
| **CF RR DeepHit Delta** | 1.093 | 1.213 | 1.174 | ❌❌❌ |
| **CF RR Exam RNN Delta** | 1.813 | 1.633 | 1.693 | ❌❌❌ |
| **CF GRU** | — | — | — | ❌ (HR=1.51) |

> [!CAUTION]
> **Nur 2 von 12 Modellen schätzen den Support-Effekt überwiegend korrekt.** Der Extended Cox Panel (alle 3 Support-Typen protektiv) und der Counterfactual HR Analyzer (2 von 3 korrekt). Die Mehrheit der Modelle – insbesondere alle RNN- und DeepHit-basierten Counterfactual-Schätzer – zeigt **schädlichen Support (RR > 1.0)**. Das ist das Dropout-Paradoxon, das die Modelle *nicht* auflösen konnten.

### C. Der Simulator V2 Ground-Truth-Vergleich — Ehrliche Bilanz

| Metrik | Simulator V2 (Ground Truth) |
| :--- | :--- |
| Dropout-Rate mit Support (Klon A) | 28.99% |
| Dropout-Rate ohne Support (Klon B) | 29.51% |
| Absolute Reduktion | **0.52 Prozentpunkte** |
| Relative Risikoreduktion | **1.75%** (RR = 0.9825) |

**Was ist vergleichbar?** Nur der **makroskopische Gesamteffekt** (alle Support-Typen kombiniert, Populations-Level). Kein einzelnes Modell liefert direkt diesen Wert, weil die Modelle **pro Support-Typ und pro Semester-Periode** schätzen. Die Behauptung in meinem früheren Artefakt, dass "DML 1.82% schätzte vs. wahre 1.75%", war **methodisch unsauber**, weil:

1. Der DML-Wert von 0.919 ist der **Mean Relative Risk für fachlichen Support allein**, nicht der kombinierte Populations-Effekt
2. Die 1.82% wurde nirgends berechnet — sie war eine von mir unzulässig extrapolierte Zahl
3. Die Granularitäten sind inkommensurabel: Macro-RR (Populations-Vergleich) vs. Micro-RR (Individuelle Hazard-Rate pro Periode)

> [!IMPORTANT]
> **Selbstkorrektur:** Mein früheres Artefakt `ground_truth_vs_models.md` enthielt cherry-picked Vergleiche und eine frei erfundene "1.82%"-Zahl. Die korrekte Darstellung ist: Der DML schätzt für fachlichen Support eine individuelle Hazard-Reduktion von ~8% (RR=0.919), was *qualitativ konsistent* mit dem makroskopischen 1.75%-Effekt ist, aber *quantitativ nicht direkt vergleichbar*.

---

## Teil III: Kritische Befunde

### 1. "Causal Forest" existiert nicht

In mehreren Dokumenten (Walkthrough, ground_truth_vs_models.md, mündliche Kommunikation) wurde der Begriff **"Causal Forest"** verwendet. Eine Codebase-Analyse zeigt:

- **Kein `econml`, `doubleml`, `causalml` oder `grf` Package** ist installiert oder im `requirements.txt`
- **Kein Causal-Forest-Algorithmus** ist implementiert
- Was existiert, ist ein **manuelles 2-Stufen-DML** in [`dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py): Logistic Regression (Propensity Score) → Keras Neural Network (Hazard auf orthogonalisierten Residuen)

Das ist methodisch eine legitime DML-Implementierung nach Chernozhukov et al. — aber es ist **kein Causal Forest**. Die Bezeichnung muss korrigiert werden, um die wissenschaftliche Integrität zu wahren.

### 2. DeepSurv ist systematisch gescheitert

Alle drei DeepSurv-Varianten (Delta, Panel, Exam) haben ROC-AUC-Werte zwischen 0.46 und 0.56. Das bedeutet:

- Die `breslow_cox_loss` Custom-Loss-Funktion konvergiert nicht zuverlässig
- Die Orakel-Analyse mit DeepSurv (Baseline: 0.52, Oracle: 0.53) ist **wertlos** als Validierung
- Nur die Logistic Hazard-Variante der Orakel-Analyse ist belastbar (0.797 → 0.807)

### 3. Die Mehrheit der Counterfactual-Modelle liefert invertierte Ergebnisse

Von 12 kontrafaktischen Analysen zeigen **10 Modelle schädlichen Support** für mindestens 2 von 3 Support-Typen. Das liegt am **nicht aufgelösten Confounding-by-Indication**: Studierende, die Support nutzen, haben bereits schlechtere Prognosen. Die meisten Modelle können diesen Selektions-Bias nicht trennen.

**Nur das Extended Cox Panel (time-varying Kovariaten) und das DML-Modell (für fachlichen Support) lösen das Paradoxon teilweise auf.**

### 4. Die Orakel-Analyse ist nur halb belastbar

| Modell-Typ | Base AUC | Oracle AUC | Lift | Bewertung |
| :--- | :---: | :---: | :---: | :--- |
| Logistic Hazard Delta | 0.797 | 0.807 | +0.010 | ✅ Belastbar, moderater aber realer Lift |
| DeepSurv Delta | 0.521 | 0.527 | +0.006 | ❌ Bedeutungslos (Basis-Modell dysfunktional) |

---

## Teil IV: Was das Projekt tatsächlich leistet — Ehrliche Stärken

Trotz der obigen Kritikpunkte bleibt das Projekt in mehreren Dimensionen beeindruckend:

### 1. Das Dropout-Paradoxon wird sichtbar gemacht und teilweise gelöst

Die Progression von "Support scheint schädlich" (statische Modelle) zu "Support ist protektiv" (Extended Cox, DML) ist das intellektuelle Rückgrat des Projekts. Dass die Mehrheit der Sequenz-Modelle das Paradoxon *nicht* auflöst, ist kein Versagen — es ist eine **ehrliche Dokumentation der Schwierigkeit kausal-inferenzieller Fragestellungen**.

### 2. Die Simulator-Architektur ist methodisch stark

- 50.000 Studierende mit realistischen Karriereverläufen
- Reaktives Confounding (+20% Support-Nutzung nach Fehlversuch)
- Kontrafaktische Notes (`note_counterfactual`)  
- Deterministisches Trajektorien-Klonen (Simulator V2)

### 3. Die Methodenbreite ist für ein Uni-Projekt außergewöhnlich

13+ Modelle über 5 Stufen (Baseline → Landmark → Panel → Sequence → Competing Risks + DML). Das ist ein systematisches Methodenportfolio, kein willkürliches Modellsammeln.

### 4. Die Ground-Truth-Analyse ist ein Unique Selling Point

Die Fähigkeit, geschätzte Effekte gegen die wahren Simulator-Mechanismen zu validieren, ist ein Vorteil synthetischer Daten, der hier bewusst genutzt wird. Der Mikro-Effekt (ATT = -0.170 Notenpunkte, Bestehensquote +4.21%) und der Makro-Effekt (1.75% Dropout-Reduktion) sind methodisch sauber berechnet.

---

## Teil V: Gesamtbewertung im Vergleich zum Portfolio Review

| Dimension | Portfolio Review (vorher) | Aktueller Stand | Δ |
| :--- | :--- | :--- | :---: |
| PH-Diagnose | ❌ Fehlte | ✅ Schoenfeld-Residuen | +++ |
| Ground-Truth-Vergleich | ❌ Fehlte | ✅ Mikro + Makro | +++ |
| Kalibrierung | ❌ Fehlte | ✅ Reliability Diagram (DTL) | ++ |
| Hidden Variables | ❌ Nicht genutzt | ✅ Panel + Oracle-Lift | ++ |
| KI-Transparenz | ❌ Fehlte | ✅ README + Stack-Auflistung | ++ |
| Dashboard | ❌ Broken | ❌ Immer noch broken | — |
| Sensitivitätsanalyse | ❌ Fehlte | ❌ Fehlt immer noch | — |
| Terminologie-Korrektheit | n/a | ❌ "Causal Forest" fälschlich | − |
| DeepSurv-Performance | Nicht evaluiert | ❌ Dysfunktional (AUC ~0.52) | − |

### Gesamteinschätzung

Das Projekt hat die drei wichtigsten methodischen Lücken aus dem Portfolio Review geschlossen (PH, Ground-Truth, Kalibration). Es hat mit dem Simulator V2 und den Orakel-Modellen genuine neue Analyseschichten hinzugefügt.

Gleichzeitig offenbart die systematische Auswertung **unbequeme Wahrheiten**, die nicht unter den Teppich gekehrt werden sollten:

1. **10 von 12 Counterfactual-Modellen scheitern** an der Auflösung des Dropout-Paradoxons
2. **DeepSurv ist als Architektur gescheitert** (breslow_cox_loss konvergiert nicht)
3. **Die Terminologie war teilweise falsch** ("Causal Forest" statt "manuelles DML")
4. **Der quantitative Ground-Truth-Vergleich war cherry-picked** (inkommensurable Granularitäten)

Diese Schwächen ehrlich zu dokumentieren würde das Projekt paradoxerweise **stärker** machen: Ein Interviewer, der sieht, dass jemand 13 Modelle trainiert hat und offen dokumentiert, dass die Mehrheit das kausale Problem nicht löst, zeigt methodische Reife.

---

## Teil VI: Empfehlungen

### Sofort umsetzbar

1. **"Causal Forest" durch "DML (Orthogonalisiertes Neuronales Netz)"** ersetzen — überall in der Dokumentation
2. **DeepSurv-Varianten als gescheitert kennzeichnen** oder aus der Haupttabelle in eine "Negative Results"-Sektion verschieben
3. **ground_truth_vs_models.md überarbeiten**: Keine Vergleiche unterschiedlicher Granularitäten, klare Trennung Mikro vs. Makro

### Für die Projektpräsentation

4. **Die "10/12 scheitern"-Erkenntnis als Feature darstellen**: "Wir haben 12 Modelle getestet; nur Extended Cox (time-varying) und DML lösen den Selektionsbias auf. Das zeigt die Notwendigkeit kausal-inferenzieller Methoden."
5. **Orakel-Analyse nur für Logistic Hazard berichten** (DeepSurv-Ergebnis ist nicht belastbar)
6. **Simulator V2 Ergebnis klar rahmen**: "Der wahre Makro-Effekt von 1.75% Dropout-Reduktion ist qualitativ konsistent mit den per-Support-Typ-Schätzungen der DML- und Cox-Modelle"

### Mittelfristig (hoher Impact)

7. **Kombiniertes kontrafaktisches RR berechnen**: Im DML-Modell *alle drei* Support-Typen gleichzeitig auf 0/1 setzen und einen kombinierten Populations-Effekt schätzen, der direkt mit dem Simulator-V2-Makro-Effekt vergleichbar wäre
8. **Sensitivitätsanalyse**: `gewicht_support_boost` zwischen 0.02 und 0.08 variieren
9. **Dashboard reparieren oder ehrlich als "nicht implementiert" markieren**
