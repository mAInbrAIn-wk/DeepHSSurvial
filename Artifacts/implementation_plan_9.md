# Implementation Plan: Systematische Erweiterung der Kontrafaktischen Analyse

**Datum:** 21. August 2026  
**Kontext:** Basierend auf den Annotationen zum [counterfactual_methods_review.md](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/counterfactual_methods_review.md)

---

## Vorbemerkungen: Ergebnisse der Code-Recherche

Vor der eigentlichen Planung fasse ich die Antworten auf die offenen Fragen zusammen:

### A. Motivation und soziale Integration wirken auch auf Noten (indirekter Kanal)

Die Leistungsformel in [`simulation_v2.py`](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v2.py#L143-L155) lautet:

$$\text{leistung} = 0{,}55 + (2{,}5 - \text{erwartete\_note}) \times 0{,}40 + \underbrace{(\text{motivation} - 0{,}5) \times \mathbf{0{,}50}}_{\text{direkt!}} + \underbrace{(\text{soz\_int} - 0{,}5) \times \mathbf{0{,}20}}_{\text{direkt!}} - \ldots$$

**Folgerung:** Überfachlicher Support (+0,10 Motivation) verbessert die Noten indirekt um $+0{,}10 \times 0{,}50 = +0{,}05$ Leistungspunkte (≈ 0,2 Notenstufen). Psychosozialer Support (+0,075 Motivation, +0,175 Soz.Int.) wirkt sogar über **beide** Kanäle: $+0{,}075 \times 0{,}50 + 0{,}175 \times 0{,}20 = +0{,}0725$ Leistungspunkte. Dieser indirekte Pfad ist im bisherigen Review unterschlagen worden.

### B. Selbstselektion unterscheidet sich zwischen Support-Typen (aus dem Code, nicht spekuliert)

Die Aufnahmewahrscheinlichkeit $p$ in [`simulation_v3.py`](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L171-L193) ist **typ-spezifisch**:

| Support-Typ | Aufnahmeformel | Selektionsmechanismus |
|:---|:---|:---|
| **Fachlich** | $p = 0{,}05 + (\text{erwartete\_note} - 2{,}0) \times 0{,}05 + 0{,}20 \times \text{Wiederholer}$ | Schwächere Studierende (höhere erwartete Note, Wiederholungsprüfungen) nehmen häufiger teil |
| **Überfachlich** | $p = 0{,}05 + (0{,}5 - \text{motivation}) \times 0{,}15$ | Studierende mit **niedriger Motivation** nehmen häufiger teil |
| **Psychosozial** | $p = 0{,}01 + (0{,}5 - \text{soz\_int}) \times 0{,}12$ | Studierende mit **niedriger sozialer Integration** nehmen häufiger teil |

**Folgerung zur psychosozial-vs.-überfachlich-Frage:** Die Spekulation im Review war falsch. Im Simulator ist die psychosoziale Selbstselektion nicht „weniger mit schlechten Leistungsindikatoren korreliert" – sie korreliert mit sozialer Integration, die überfachliche mit Motivation. Beide sind latent. Aber psychosozialer Support hat den **deutlich stärkeren Effekt** (+0,175 Soz.Int. vs. +0,05) und **niedrigere Basisrate** ($p_{\text{base}} = 0{,}01$ vs. $0{,}05$), was den Selektionsbias möglicherweise weniger ausgeprägt macht.

### C. DeepSurv Panel sieht `cum_cp`

Die Feature-Tabelle im Review war an dieser Stelle falsch. Das DeepSurv Panel Modell sieht: `hzb_note`, `erwerbstaetigkeit_std`, `t_stop`, `t_start`, **`cum_cp`**, **`cum_fails`**, `stg_name`, `erstakademiker`, plus die 3 Treatment-Variablen. CP-Informationen sind also vorhanden.

### D. DML verwendet ebenfalls partielle Isolation

Die Code-Analyse zeigt: DML setzt beim kontrafaktischen Toggle nur den Residual-Wert des **Ziel-Supports** auf $1 - \hat{e}$ bzw. $-\hat{e}$. Die anderen beiden Support-Residuals bleiben bei ihren **beobachteten** Werten. DML verhält sich also wie DeepSurv (partiell), nicht wie Logistic Hazard/DeepHit (reine Isolation).

### E. Oracle-Modelle: Latente Variablen kaum informativ

| Modell | Baseline AUC | Oracle AUC | Lift |
|:---|:---:|:---:|:---:|
| Logistic Hazard Delta | 0,7665 | 0,7756 | **+0,91%** |
| DeepSurv Delta | 0,5318 | 0,5318 | **≈ 0,00%** |

Die latenten Variablen (`motivation`, `soz_integration`, `erwartete_note`) liefern, wenn direkt beobachtbar, **kaum zusätzliche Vorhersagekraft**. Das bestätigt, dass die Modelle die latenten Variablen aus ihren verrauschten Auswirkungen (Noten, Fehlversuche, CP-Rückstand) bereits recht gut rekonstruieren können. Für die **Vorhersage** reicht das – für die **kausale Attribution** offenbar nicht, da die Zuordnung „welcher Kanal hat gewirkt" trotzdem verloren geht.

---

## 1. Neue Simulations-Universen F, G, H (Reine Isolation)

### Ziel
Ergänzung der 5 bestehenden Universen um 3 weitere, in denen **nur ein einziger** Support-Typ aktiv und die anderen beiden blockiert sind:

| Universum | `block_fach` | `block_uebf` | `block_psych` | Beschreibung |
|:---|:---:|:---:|:---:|:---|
| **F** | ✗ | ✓ | ✓ | **Nur fachlicher Support** |
| **G** | ✓ | ✗ | ✓ | **Nur überfachlicher Support** |
| **H** | ✓ | ✓ | ✗ | **Nur psychosozialer Support** |

### Ground-Truth-Vergleichspaare

| Vergleich | Kontroll-Universum | Treatment-Universum | Interpretation |
|:---|:---|:---|:---|
| **Partielle Wirkung (bisherig)** | A (alle aktiv) | C / D / E (eines blockiert) | „Wie viel geht verloren, wenn man diesen Support wegnimmt?" |
| **Isolierte Wirkung (neu)** | B (keiner aktiv) | F / G / H (nur einer aktiv) | „Wie viel gewinnt man, wenn man nur diesen Support hinzufügt?" |

### Änderungen

#### [MODIFY] [`simulation_v3.py`](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v3.py)
- Die Funktion `simuliere_kohorte()` akzeptiert bereits `block_fach`, `block_uebf`, `block_psych`. Keine Änderung an der Kernlogik nötig.

#### [MODIFY] [`run_overnight.py`](file:///c:/GitHub_public/Abschlussprojekt/src/run_overnight.py) oder dediziertes Launcher-Skript
- Ergänze die Universum-Konfiguration um F, G, H:
  ```python
  universes = {
      "A": {"block_fach": False, "block_uebf": False, "block_psych": False},
      "B": {"block_fach": True,  "block_uebf": True,  "block_psych": True},
      "C": {"block_fach": True,  "block_uebf": False, "block_psych": False},
      "D": {"block_fach": False, "block_uebf": True,  "block_psych": False},
      "E": {"block_fach": False, "block_uebf": False, "block_psych": True},
      "F": {"block_fach": False, "block_uebf": True,  "block_psych": True},   # NEU
      "G": {"block_fach": True,  "block_uebf": False, "block_psych": True},   # NEU
      "H": {"block_fach": True,  "block_uebf": True,  "block_psych": False},  # NEU
  }
  ```

#### [MODIFY] Makro-Effekt-Berechnung
- Erweitere die Ground-Truth-RR-Berechnung um die neuen isolierten Vergleiche:
  - $\text{RR}_{\text{fach,isoliert}} = R_B / R_F$
  - $\text{RR}_{\text{uebf,isoliert}} = R_B / R_G$
  - $\text{RR}_{\text{psych,isoliert}} = R_B / R_H$
- Speichere Ergebnisse in `true_macro_effects_v3_extended.json`

> [!IMPORTANT]
> **Laufzeit:** Die Simulation eines Universums dauert ca. 5–8 Minuten. Die 3 neuen Universen (F, G, H) benötigen daher ca. 15–25 Minuten zusätzlich. Da die Studierenden-Population identisch ist (gleicher `POPULATION_SEED`), müssen nur die Semesterverläufe neu simuliert werden.

---

## 2. Zwei parallele Teststrategien für alle Counterfactual-Modelle

### Ziel
Jedes Counterfactual-Modell soll **zwei** RR/HR-Schätzer liefern:

1. **Partieller Schätzer** (≙ A vs. C/D/E): Nur der Ziel-Support wird getoggelt, alle anderen bleiben bei ihrem **beobachteten Wert**
2. **Isolierter Schätzer** (≙ B vs. F/G/H): Alle 3 Supports auf 0 in der Kontrolle, nur der Ziel-Support auf 1 im Treatment

### Betroffene Skripte und Änderungen

#### Panel-Modelle (Person-Semester-Ebene)

| Skript | Aktuell | Änderung |
|:---|:---|:---|
| [`counterfactual_hr_analyzer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_analyzer.py) (DeepSurv Panel) | Partiell | **+ Isolierten Schätzer hinzufügen** |
| [`counterfactual_hr_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_delta.py) (DeepSurv Delta) | Partiell | **+ Isolierten Schätzer hinzufügen** |
| [`counterfactual_rr_logistic_hazard_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py) (Logistic Hazard) | Isoliert | **+ Partiellen Schätzer hinzufügen** |
| [`dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py) (DML) | Partiell | **+ Isolierten Schätzer hinzufügen** |

#### Sequenz-Modelle (Semester-/Prüfungsebene)

| Skript | Aktuell | Änderung |
|:---|:---|:---|
| [`counterfactual_rr_deephit_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_deephit_delta.py) (DeepHit) | Isoliert | **+ Partiellen Schätzer hinzufügen** |
| [`counterfactual_inference_semester_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py) (Transformer) | Partiell, nur Fachlich | **+ Isolierten Schätzer + alle 3 Support-Typen** |
| [`counterfactual_rr_exam_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_exam_rnn_delta.py) (Exam RNN Delta) | Isoliert | **+ Partiellen Schätzer hinzufügen** |
| [`counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py) (GRU V2) | Isoliert | **+ Partiellen Schätzer hinzufügen** |

#### Implementierungsmuster (für jedes Skript gleichartig)

```python
for supp_col, label in support_types:
    # --- Variante A: PARTIELLER Schätzer (≙ A vs. C/D/E) ---
    # "Was passiert, wenn wir NUR diesen Support wegnehmen?"
    control_partial = test_data.copy()
    treated_partial = test_data.copy()
    control_partial[supp_col] = 0.0       # Ziel-Support AUS
    treated_partial[supp_col] = 1.0       # Ziel-Support AN
    # Andere Supports: BEOBACHTETER Wert (unverändert)
    
    hr_partial = compute_hr(model, control_partial, treated_partial)
    
    # --- Variante B: ISOLIERTER Schätzer (≙ B vs. F/G/H) ---
    # "Was bringt NUR dieser Support, wenn sonst nichts aktiv ist?"
    control_isolated = test_data.copy()
    treated_isolated = test_data.copy()
    for col in all_support_cols:
        control_isolated[col] = 0.0       # ALLE Supports AUS
        treated_isolated[col] = 0.0       # ALLE Supports AUS
    treated_isolated[supp_col] = 1.0      # NUR Ziel-Support AN
    
    hr_isolated = compute_hr(model, control_isolated, treated_isolated)
```

### Output-Format (erweiterte Metriken-JSON)

Jedes Skript speichert künftig:
```json
{
  "fach_partial":  {"mean_hr": ..., "median_hr": ..., "q05": ..., "q95": ...},
  "fach_isolated": {"mean_hr": ..., "median_hr": ..., "q05": ..., "q95": ...},
  "uebf_partial":  {"mean_hr": ..., "median_hr": ..., "q05": ..., "q95": ...},
  "uebf_isolated": {"mean_hr": ..., "median_hr": ..., "q05": ..., "q95": ...},
  "psych_partial":  {"mean_hr": ..., "median_hr": ..., "q05": ..., "q95": ...},
  "psych_isolated": {"mean_hr": ..., "median_hr": ..., "q05": ..., "q95": ...}
}
```

---

## 3. Erweiterung des Semester-Transformers

### Aktueller Zustand
[`counterfactual_inference_semester_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py) evaluiert nur `fach_supp_cum` (Spalte 3). Überfachlich und psychosozial werden nicht kontrafaktisch untersucht.

### Änderung
Erweiterung auf alle 3 Support-Kanäle in beiden Varianten:

| Feature-Index | Feature-Name | Partiell | Isoliert |
|:---:|:---|:---:|:---:|
| 3 | `fach_supp_cum` | ✓ | ✓ |
| 4 | `uebf_supp_cum` | ✓ (NEU) | ✓ (NEU) |
| 5 | `psych_supp_cum` | ✓ (NEU) | ✓ (NEU) |

---

## 4. Sequenzlängen-Leakage: Analyse & Empfehlung

### Befund aus der Code-Recherche

| Modell | Masking-Mechanismus | Max Seq | Leakage-Risiko |
|:---|:---|:---:|:---|
| **Deep Exam-Transformer Survival** | Custom `AttentionPooling` mit $-10^9$ auf Padding-Logits; **keine** Keras-`Masking`-Layer | 40 | ⚠️ **Hoch**: Multi-Head-Attention-Blöcke erhalten keine `attention_mask`, nur die finale Pooling-Schicht maskiert. Backbone-Attention kann Padding-Tokens „sehen" und daraus die Sequenzlänge ableiten. |
| **Recurrent Exam GRU (Base & V2)** | Keras `Masking(mask_value=-99)` + `masked_binary_crossentropy` | 50 | ✅ **Niedrig**: `Masking`-Layer propagiert Boolean-Maske durch GRU; Hidden State wird bei Padding-Steps nicht aktualisiert. |
| **Recurrent Exam GRU (timeseries_exam)** | Keras `Masking(mask_value=-99)` mit `return_sequences=False` | dynamisch | ✅ **Niedrig**: Identischer Mechanismus. |

### Empfehlung

Die Multi-Head-Attention-Blöcke im Deep Exam-Transformer sollten eine explizite `attention_mask` erhalten, die Padding-Tokens auch im Backbone ausmaskiert – nicht erst in der finalen Pooling-Schicht. Zusätzlich könnte eine Variante mit Keras-`Masking`-Layer getestet werden, um die Ergebnisse mit den RNN-Modellen direkt vergleichbar zu machen.

> [!NOTE]
> Das aktuelle ROC-AUC von 0,9999 ist mit hoher Wahrscheinlichkeit durch dieses Sequenzlängen-Leakage verursacht und **nicht** als echte Modellleistung zu werten. Nach Behebung des Leakage sollte die Performance in den Bereich der RNN-Modelle (ROC-AUC ≈ 0,85–0,87) fallen.

---

## 5. Korrektur und Erweiterung des Methods-Review

### Zu korrigierende Punkte im bestehenden Review-Dokument:

1. **Indirekter Noteneffekt:** Überfachlicher und psychosozialer Support wirken auch indirekt auf Noten (über Motivation × 0,50 und Soz.Int. × 0,20 in der Leistungsformel). Der Review-Text „Noteneffekt: keiner" und „hinterlässt keine beobachtbare Spur" ist falsch.

2. **Feature-Tabelle korrigieren:** DeepSurv Panel sieht `cum_cp` und `cum_fails` (nicht „–" wie in der Tabelle). `hzb_note` und `erwerbstaetigkeit_std` als „statisch" kennzeichnen.

3. **DML-Isolation korrigieren:** DML verwendet ebenfalls partielle Isolation (nicht reine Isolation), genau wie DeepSurv.

4. **Spekulation über psychosozialen Selektionsbias ersetzen:** Durch die aus dem Code abgeleiteten tatsächlichen Unterschiede in den Aufnahmeformeln (Motivation-getrieben vs. Soz.Int.-getrieben, verschiedene Basisraten).

5. **Oracle-Befund einordnen:** Die minimale Oracle-Lift (+0,91% für Logistic Hazard) bestätigt, dass die latenten Variablen aus den beobachtbaren Features gut rekonstruierbar sind – für Vorhersage, aber nicht für kausale Attribution.

---

## 6. Zusammenfassende Ergebnistabelle (Zielformat)

Nach Durchführung aller Schritte soll eine finale Vergleichstabelle wie folgt aussehen:

| Modell | Fach (partiell) | Fach (isoliert) | Übf. (partiell) | Übf. (isoliert) | Psych (partiell) | Psych (isoliert) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ground Truth** | $R_A/R_C$ | $R_B/R_F$ | $R_A/R_D$ | $R_B/R_G$ | $R_A/R_E$ | $R_B/R_H$ |
| Extended Cox Delta | HR | — | HR | — | HR | — |
| DeepSurv Panel | HR | HR | HR | HR | HR | HR |
| DeepSurv Delta | HR | HR | HR | HR | HR | HR |
| Logistic Hazard Delta | RR | RR | RR | RR | RR | RR |
| DML Orthogonal | RR | RR | RR | RR | RR | RR |
| DeepHit Delta | RR | RR | RR | RR | RR | RR |
| Semester Transformer | HR | HR | HR | HR | HR | HR |
| Exam RNN Delta | RR | RR | RR | RR | RR | RR |
| Exam GRU V2 | RR | RR | RR | RR | RR | RR |

> [!NOTE]
> Der Extended Cox Delta (statistisches Modell) liefert eine globale parametrische HR; eine isolierte Variante ist dort nur möglich, wenn man die anderen beiden Support-Variablen in den Daten auf 0 setzt und das Modell neu schätzt. Dies wäre eine dritte Option – oder man akzeptiert, dass die Cox-HR konzeptionell „partiell, adjustiert" ist.

---

## Verification Plan

### Automatisierte Tests
1. Simulation der 3 neuen Universen F, G, H mit konsistenter RNG-Synchronisation
2. Plausibilitätsprüfung: $R_F$ sollte zwischen $R_A$ und $R_B$ liegen (nur ein Support-Typ aktiv ≈ weniger Schutz als alle, aber mehr als keiner)
3. Berechnung und Vergleich: $\text{RR}_{\text{partiell}}$ vs. $\text{RR}_{\text{isoliert}}$ für alle Modelle

### Manuelle Verifikation
1. Stichprobenartige Prüfung der kontrafaktischen Tensor-Manipulation in 2–3 Sequenzmodellen
2. Sichtprüfung der erweiterten Metriken-JSONs auf Vollständigkeit
3. Finale Ergebnistabelle im Walkthrough

---

## Arbeitsreihenfolge

1. Simulation F, G, H durchführen (~20 Min.)
2. Ground-Truth-RR berechnen (partiell + isoliert)
3. Counterfactual-Skripte erweitern (8 Skripte × 2 Varianten)
4. Semester-Transformer auf alle 3 Support-Typen erweitern
5. Alle erweiterten Counterfactual-Analysen ausführen
6. Methods-Review korrigieren und erweitern
7. Finale Vergleichstabelle erstellen
