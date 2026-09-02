# Aktualisierter Änderungsbericht: Codebase Refactoring & Archivierung (Phase 1)

Deine Einwände sind ein Volltreffer. Ich habe den Code der `metrics_logger.py` und der `run_feature_grid_experiments.py` gerade noch einmal tief analysiert. 

Du hast vollkommen recht: Der Grid-Runner durchbricht aktuell das Paradigma komplett, weil er (auf über 500 Zeilen) die Keras-Architekturen einfach **hartcodiert per Copy-Paste** in sich selbst definiert, anstatt sie aus den Modelldateien zu importieren! Und der Metrics-Logger nimmt einfach dumme Strings (wie `f"{model}_{mode}"`) entgegen, statt die Metadaten sauber zu strukturieren.

Hier ist der entsprechend **gehärtete Plan** für die Umsetzung:

## 1. Das "Strict Wrapper" Paradigma (Der neue Workflow)

```text
src/deepsupport/
├── data_engine/     (feature_builder.py etc.)
├── models/          (JEDE Architektur hat eine Datei mit `build_model()` und `train()`)
├── runners/         (Grid-Runner importiert `build_model` und steuert die Schleifen)
└── evaluation/      (Der NEUE metrics_logger.py)
```

## 2. Refactoring der Kern-Komponenten (Deine Punkte)

### A. Der neue Metrics Logger (`metrics_logger.py`)
Er wird komplett überarbeitet, um Parameter "durchzureichen", statt sie in Dateinamen-Strings zu verstecken.
* **Neu:** `save_metrics(architecture_name, mode, dataset_scenario, metrics_dict, base_dir)`
* Er erzwingt ein **einheitliches JSON-Schema** (immer `roc_auc`, `pr_auc`, `brier` – auch wenn sie 0.0 sind), sodass asymmetrische Daten (wie wir sie bei S01 gesehen haben) künftig unmöglich sind.

### B. Reparatur des Grid-Runners (`run_feature_grid_experiments.py`)
Dieses 500-Zeilen-Skript wird radikal gekürzt. Es darf künftig **keine einzige Keras-Schicht** mehr selbst definieren!
* **Ablauf künftig:** Der Runner iteriert über `MODES`. Für jeden Modus importiert er z.B. `build_exam_gru()` aus `models/exam_gru.py`, füttert es mit den Feature-Builder-Daten, und übergibt die Ergebnisse an den neuen `metrics_logger`. 
* So operiert der Grid-Runner wirklich auf einer reinen "Verteiler-Ebene", wie Du es intuitiv gefordert hast.

### C. Die MLPs / Baselines
Da Du (völlig richtig) angemerkt hast, dass die scikit-learn Modelle rasend schnell sind, brechen wir diese nicht in 10 Mini-Skripte auf. 
Wir fassen sie in **`models/baseline_classifiers.py`** und **`models/baseline_regressors.py`** zusammen. Sie stellen Funktionen wie `run_random_forest()`, `run_svm()`, `run_mlp()` bereit, die vom Runner bequem aufgerufen werden können.

## 3. Causal Inference & Kontrafaktik (Sicherheitsnetz)

Wie im Deep-Dive-Audit besprochen, werde ich die 16 `counterfactual_*.py` Skripte **nicht archivieren**. Sie ziehen nach `evaluation/causal/` um.
Ich werde ein dediziertes Such-und-Ersetz-Skript schreiben, das die hartcodierten Lade-Befehle (z. B. `load_model("timeseries_exam_transformer.keras")`) in diesen Skripten exakt auf die neuen Dateinamen (`exam_transformer.keras`) umschreibt, damit Deine Kausalkette nach dem Umbenennen sofort fehlerfrei weiterläuft.

---

## Umsetzung

Wir haben jetzt ein wasserdichtes Fundament, das sowohl Architektur-Sauberkeit als auch die Rettung der Kontrafaktik-Inferenz garantiert. 

Mit diesem finalisierten Paradigma: Darf ich nun die Umstrukturierung auf dem Dateisystem beginnen?
