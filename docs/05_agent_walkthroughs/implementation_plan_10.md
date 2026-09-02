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
Jedes Counterfactual-Modell soll **zwei** RR/HR-Schätzer liefern, die verschiedene kontrafaktische Fragen beantworten:

1. **Partieller Schätzer** (≙ A vs. C/D/E): Ziel-Support auf 0 (Kontrolle) vs. **beobachteter Wert** (Treatment). Andere Supports bleiben beobachtet. → „Was verliert man, wenn man diesen Support wegnimmt?"
2. **Isoliert realistischer Schätzer** (≙ B vs. F/G/H): Alle 3 Supports auf 0 (Kontrolle) vs. nur Ziel-Support **beobachtet** (Treatment), andere auf 0. → „Was bringt nur dieser Support in beobachteter Dosis?"

| Variante | Kontrolle | Treatment | GT-Vergleich | Frage |
|:---|:---|:---|:---|:---|
| **Partiell** | Ziel=0, andere=beobachtet | Ziel=**beobachtet**, andere=beobachtet | A vs. C/D/E | Effekt des Wegnehmens |
| **Isoliert realistisch** | Alle 3 = 0 | Ziel=**beobachtet**, andere=0 | B vs. F/G/H | Reiner Einzeleffekt (realist. Dosis) |

> [!IMPORTANT]
> **Mit Zählvariablen statt binären Indikatoren** (siehe Sektion 5) wird „0" zu „keine Teilnahme" und der beobachtete Wert bildet die tatsächliche Dosierung ab. Die forcierten Varianten (Ziel = 1) entfallen, weil „1 Teilnahme" keine natürliche Interventionsstärke darstellt.

### Betroffene Skripte und Änderungen

#### Panel-Modelle (Person-Semester-Ebene)

| Skript | Aktuell | Änderung |
|:---|:---|:---|
| [`counterfactual_hr_analyzer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_analyzer.py) (DeepSurv Panel) | Forciert (binär) | **Umstellen auf Zählung + 2 Varianten** |
| [`counterfactual_hr_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_delta.py) (DeepSurv Delta) | Forciert (binär) | **Umstellen auf Zählung + 2 Varianten** |
| [`counterfactual_rr_logistic_hazard_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py) (Logistic Hazard) | Isoliert (binär) | **Umstellen auf Zählung + 2 Varianten** |
| [`dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py) (DML) | Forciert (binär) | **Umstellen auf Zählung + 2 Varianten** |

#### Sequenz-Modelle (Semester-/Prüfungsebene)

| Skript | Aktuell | Änderung |
|:---|:---|:---|
| [`counterfactual_rr_deephit_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_deephit_delta.py) (DeepHit) | Isoliert (binär) | **Umstellen auf Zählung + 2 Varianten** |
| [`counterfactual_inference_semester_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py) (Transformer) | Forciert (binär), nur Fachlich | **Umstellen auf Zählung + 2 Varianten × 3 Support-Typen** |
| [`counterfactual_rr_exam_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_exam_rnn_delta.py) (Exam RNN Delta) | Isoliert (binär) | **Umstellen auf Zählung + 2 Varianten** |
| [`counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py) (GRU V2) | Isoliert (binär) | **Umstellen auf Zählung + 2 Varianten** |

#### Implementierungsmuster (für jedes Skript gleichartig)

```python
# supp_cols: Liste der Zähl-Features für den jeweiligen Support-Typ
# z.B. fach_cols = ['fach_supp_count'] (Panel) oder ['fach_expo_vorher', 'fach_expo_glz'] (Exam)
for supp_cols, label in support_types:
    # --- Variante 1: PARTIELLER Schätzer (≙ A vs. C/D/E) ---
    # "Was verliert man, wenn man diesen Support wegnimmt?"
    control_partial = test_data.copy()
    treated_partial = test_data.copy()           # BEOBACHTETER Wert bleibt stehen
    for col in supp_cols:
        control_partial[col] = 0                 # NUR Ziel-Support AUS (Zählung = 0)
    # Andere Supports: BEOBACHTETER Wert (unverändert in beiden)
    
    hr_partial = compute_hr(model, control_partial, treated_partial)
    
    # --- Variante 2: ISOLIERT REALISTISCH (≙ B vs. F/G/H) ---
    # "Was bringt NUR dieser Support in beobachteter Dosis?"
    control_isolated = test_data.copy()
    treated_isolated = test_data.copy()
    for col in all_support_cols:
        control_isolated[col] = 0                # ALLE Supports AUS
        treated_isolated[col] = 0                # ALLE Supports AUS
    for col in supp_cols:
        treated_isolated[col] = test_data[col]   # NUR Ziel-Support = BEOBACHTET
    
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
Erweiterung auf alle 3 Support-Kanäle in allen 3 Varianten (partiell, isoliert, forciert):

| Feature-Index | Feature-Name | Partiell | Isoliert | Forciert |
|:---:|:---|:---:|:---:|:---:|
| 3 | `fach_supp_cum` | ✓ | ✓ | ✓ |
| 4 | `uebf_supp_cum` | ✓ (NEU) | ✓ (NEU) | ✓ (NEU) |
| 5 | `psych_supp_cum` | ✓ (NEU) | ✓ (NEU) | ✓ (NEU) |

---

## 4. Deep Exam-Transformer Survival: Zwei Architektur-Varianten

### Problem (detailliert erläutert in [sequence_length_and_censoring.md](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/sequence_length_and_censoring.md))

Der Deep Exam-Transformer Survival ist **kein** sequenzieller Hazard-Schätzer, sondern ein **statischer binärer Klassifikator**, der die gesamte Prüfungshistorie sieht und eine einzige Dropout-Wahrscheinlichkeit ausgibt. Die Multi-Head-Attention-Blöcke erhalten keine `causal_mask`, wodurch die Sequenzlänge (Absolventen: ∅ 18,7 Exams, Abbrecher: ∅ 10,7) trivial ablesbar ist → ROC-AUC = 0,9999 (Artefakt).

### Option A: Umbau zum sequenziellen Hazard-Modell (Hauptvariante)

#### [MODIFY] [`deep_transformer_regression.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_transformer_regression.py)

**Backbone:** `use_causal_mask=True` in allen Multi-Head-Attention-Blöcken:
```python
attn_output = MultiHeadAttention(
    num_heads=num_heads, key_dim=d_model // num_heads, dropout=0.1
)(query=x, value=x, key=x, use_causal_mask=True)
```

**Output:** `AttentionPooling` → `Dense(1, sigmoid)` ersetzen durch:
```python
outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)
```

**Loss:** `binary_crossentropy` ersetzen durch `masked_binary_crossentropy` (identisch zu den RNN-Modellen).

**Target:** Von `y_surv` (statisch, eine Zahl pro Student) umbauen zu `y_seq` (Shape `(N, max_exams, 1)`, Event=1 nur am letzten Schritt für Dropouts, Padding=-99).

**Masking:** Keras `Masking(mask_value=-99)` als erste Input-Schicht hinzufügen.

**Modellname:** `deep_exam_transformer_causal_survival` (im Script-Registry als neues Modell).

### Option B: Masking-basierter statischer Klassifikator (experimentell)

#### [NEW] Neue Build-Funktion in [`deep_transformer_regression.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_transformer_regression.py)

Behält das statische Klassifikator-Design (AttentionPooling → Dense(1, sigmoid)), eliminiert aber das Leakage:

**Masking:** Keras `Masking(mask_value=-99)` als erste Input-Schicht.

**Backbone-Attention-Mask:** Explizite `attention_mask` aus der Masking-Schicht propagieren und an alle Multi-Head-Attention-Blöcke übergeben:
```python
mask = Masking(mask_value=PADDING_VALUE)(inputs)
# Compute boolean mask from input
padding_mask = tf.reduce_any(tf.not_equal(inputs, PADDING_VALUE), axis=-1)
# Pass to MHA
attn_output = MultiHeadAttention(...)(
    query=x, value=x, key=x, attention_mask=padding_mask[:, tf.newaxis, :]
)
```

**Output:** Beibehaltung von AttentionPooling → Dense(1, sigmoid) (eine Vorhersage pro Student).

**Modellname:** `deep_exam_transformer_masked_survival` (im Script-Registry als experimentelles Modell).

### Erwartetes Ergebnis beider Varianten
ROC-AUC sollte von 0,9999 auf den Bereich der vergleichbaren RNN-Modelle fallen (≈ 0,85–0,87).

---

## 5. Support-Exposition: Von binären Indikatoren zu Zählvariablen

> [!IMPORTANT]
> Detaillierte Analyse und Begründung: [support_exposition_empfehlung.md](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/support_exposition_empfehlung.md)

### Problem
Die Simulation modelliert **lineare Dosis-Wirkungs-Beziehungen** (z.B. 3 überfachliche Teilnahmen = +0,30 Motivation statt +0,10). Binäre Indikatoren (0/1) verschlucken diese Information vollständig.

### Maßnahme: Binäre Features durch Zählvariablen ersetzen

#### [MODIFY] Panel-Modelle: [`extended_cox_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_delta.py), [`extended_deep_survival_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_deep_survival_delta.py), [`extended_deep_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_deep_survival.py)

| Alt | Neu | Wertebereich |
|:---|:---|:---|
| `fach_supp_active` (0/1) | `fach_supp_count` | 0, 1, 2, 3 |
| `uebf_supp_active` (0/1) | `uebf_supp_count` | 0, 1, 2, 3 |
| `psych_supp_active` (0/1) | `psych_supp_count` | 0, 1, 2, 3 |

Semester-lokal; Kumulation dem Modell überlassen.

#### [MODIFY] Exam-Level Sequenzmodelle: [`recurrent_exam_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_exam_survival.py), [`recurrent_exam_survival_v2.py`](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_exam_survival_v2.py), [`deep_transformer_regression.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_transformer_regression.py)

| Alt | Neu (2 Features pro Typ) | Quelle |
|:---|:---|:---|
| `fach_supp_cum` (0/1) | `fach_expo_vorher` + `fach_expo_glz` | `agg_pruefungen.csv` |
| `uebf_supp_cum` (0/1) | `uebf_expo_vorher` + `uebf_expo_glz` | `agg_pruefungen.csv` |
| `psych_supp_cum` (0/1) | `psych_expo_vorher` + `psych_expo_glz` | `agg_pruefungen.csv` |

Vorher/gleichzeitig separat; Aggregation dem Modell überlassen.

#### [MODIFY] Semester-Level Sequenzmodelle: [`recurrent_survival_model.py`](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_survival_model.py), [`transformer_survival_model.py`](file:///c:/GitHub_public/Abschlussprojekt/src/transformer_survival_model.py)

| Alt | Neu | Quelle |
|:---|:---|:---|
| `cum_fach` (0/1, kumulativ) | `sem_fach_relevant` + `sem_fach_sonst` | `timeseries_semester.py` |
| `cum_uebf` (0/1, kumulativ) | `sem_uebf_count` | `timeseries_semester.py` |
| `cum_psych` (0/1, kumulativ) | `sem_psych_count` | `timeseries_semester.py` |

Semester-lokal; keine Kumulation (Sequenzmodelle lernen Zeitabhängigkeit selbst).

### Auswirkung auf Counterfaktische Toggle-Varianten

Mit Zählvariablen werden die **forcierten** Varianten (Ziel = 1) semantisch leer, weil „1 Teilnahme" keine natürliche Interventionsstärke mehr darstellt. Es bleiben **zwei** sinnvolle Varianten:

| Variante | Control | Treatment | GT-Vergleich |
|:---|:---|:---|:---|
| **Partiell** | Ziel = 0, andere = beobachtet | Ziel = **beobachtet**, andere = beobachtet | A vs. C/D/E |
| **Isoliert realistisch** | Alle = 0 | Ziel = **beobachtet**, andere = 0 | B vs. F/G/H |

---

## 6. Korrektur und Erweiterung des Methods-Review

### Zu korrigierende Punkte im bestehenden Review-Dokument:

1. **Indirekter Noteneffekt:** Überfachlicher und psychosozialer Support wirken auch indirekt auf Noten (über Motivation × 0,50 und Soz.Int. × 0,20 in der Leistungsformel). Der Review-Text „Noteneffekt: keiner" und „hinterlässt keine beobachtbare Spur" ist falsch.

2. **Feature-Tabelle korrigieren:** DeepSurv Panel sieht `cum_cp` und `cum_fails` (nicht „–" wie in der Tabelle). `hzb_note` und `erwerbstaetigkeit_std` als „statisch" kennzeichnen.

3. **DML-Isolation korrigieren:** DML verwendet ebenfalls partielle Isolation (nicht reine Isolation), genau wie DeepSurv.

4. **Spekulation über psychosozialen Selektionsbias ersetzen:** Durch die aus dem Code abgeleiteten tatsächlichen Unterschiede in den Aufnahmeformeln (Motivation-getrieben vs. Soz.Int.-getrieben, verschiedene Basisraten).

5. **Oracle-Befund einordnen:** Die minimale Oracle-Lift (+0,91% für Logistic Hazard) bestätigt, dass die latenten Variablen aus den beobachtbaren Features gut rekonstruierbar sind – für Vorhersage, aber nicht für kausale Attribution.

---

## 7. Oracle-Modelle: Aktualitätsprüfung

### Befund
Die Oracle-Modelle in [`train_oracle_models.py`](file:///c:/GitHub_public/Abschlussprojekt/src/train_oracle_models.py) lesen aus `output_dl/` und verwenden `build_delta_panel(data_dir)`. Die gespeicherten Metriken in `oracle_lift.json` sind identisch mit denen in `output_dl_v3.2_carryover/`, was darauf hindeutet, dass sie auf den V3.2-Daten (nicht V3.3) laufen.

### Maßnahme
Oracle-Modelle nach der Simulation von F, G, H **erneut** auf den aktuellen V3.3-Daten trainieren und die Lift-Werte aktualisieren.

---

## 8. Klarstellung: Beobachtete Werte im Counterfactual-Toggle

Die zwei Varianten unterscheiden sich darin, wie der Ziel-Support im Treatment gesetzt wird:

| Variante | Control: Ziel-Support | Treatment: Ziel-Support | Control: Andere | Treatment: Andere |
|:---|:---|:---|:---|:---|
| **Partiell** | 0 | **beobachtet** (Zählung) | beobachtet | beobachtet |
| **Isoliert realistisch** | 0 | **beobachtet** (Zählung) | 0 | 0 |

Mit Zählvariablen bedeutet „0" wirklich „keine Teilnahme" und der beobachtete Wert (z.B. 2) bildet die tatsächliche Dosis ab. Die Diskrepanz zum Ground-Truth-Universum (wo Support nur „verfügbar" ist, nicht deterministisch genutzt) wird dadurch minimiert.

---

## 9. Zusammenfassende Ergebnistabelle (Zielformat)

Nach Durchführung aller Schritte soll eine finale Vergleichstabelle wie folgt aussehen:

| Modell | Fach (part.) | Fach (isol.) | Übf. (part.) | Übf. (isol.) | Psych (part.) | Psych (isol.) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ground Truth** | $R_A/R_C$ | $R_B/R_F$ | $R_A/R_D$ | $R_B/R_G$ | $R_A/R_E$ | $R_B/R_H$ |
| Extended Cox Delta | HR | HR | HR | HR | HR | HR |
| DeepSurv Panel | HR | HR | HR | HR | HR | HR |
| DeepSurv Delta | HR | HR | HR | HR | HR | HR |
| Logistic Hazard Delta | RR | RR | RR | RR | RR | RR |
| DML Orthogonal | RR | RR | RR | RR | RR | RR |
| DeepHit Delta | RR | RR | RR | RR | RR | RR |
| Semester Transformer | HR | HR | HR | HR | HR | HR |
| Exam RNN Delta | RR | RR | RR | RR | RR | RR |
| Exam GRU V2 | RR | RR | RR | RR | RR | RR |
| Exam Transformer Causal (NEU) | RR | RR | RR | RR | RR | RR |
| Exam Transformer Masked (NEU) | RR | RR | RR | RR | RR | RR |

---

## Verification Plan

### Automatisierte Tests
1. Simulation der 3 neuen Universen F, G, H mit konsistenter RNG-Synchronisation
2. Plausibilitätsprüfung: $R_F$ sollte zwischen $R_A$ und $R_B$ liegen
3. Berechnung und Vergleich: $\text{RR}_{\text{partiell}}$ vs. $\text{RR}_{\text{isoliert}}$ für alle Modelle
4. Deep Exam-Transformer: ROC-AUC nach Umbau (beide Varianten) sollte im Bereich 0,85–0,87 liegen
5. Zählvariablen-Plausibilität: Verteilung der neuen Features prüfen (Histogramme)

### Manuelle Verifikation
1. Stichprobenartige Prüfung der kontrafaktischen Tensor-Manipulation in 2–3 Sequenzmodellen
2. Sichtprüfung der erweiterten Metriken-JSONs auf Vollständigkeit (6 Werte pro Modell: 3 Support-Typen × 2 Varianten)
3. Finale Ergebnistabelle im Walkthrough

---

## Arbeitsreihenfolge

1. **Support-Features refactorn:** Binäre Indikatoren durch Zählvariablen in allen Dataset-Buildern ersetzen
2. **Alle Modelle neu trainieren** (wegen geänderten Features)
3. **Simulation F, G, H** partiell durchführen (nur neue Universen)
4. **Ground-Truth-RR** berechnen (partiell + isoliert realistisch)
5. **Deep Exam-Transformer** umbauen: Option A (Causal Hazard) + Option B (Masked Static) und neu trainieren
6. **Counterfactual-Skripte erweitern** (8 Skripte × 2 Varianten mit Zählvariablen)
7. **Semester-Transformer** auf alle 3 Support-Typen erweitern
8. **Alle erweiterten Counterfactual-Analysen** ausführen
9. **Oracle-Modelle** auf aktuellem V3.3-Datensatz neu trainieren
10. **Methods-Review** korrigieren und erweitern
11. **Script-Registry** aktualisieren (2 neue Modelle, neue Feature-Beschreibungen)
12. **Finale Vergleichstabelle** erstellen
