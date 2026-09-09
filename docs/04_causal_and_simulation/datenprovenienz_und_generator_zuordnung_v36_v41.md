---
created: 2026-09-07
last_updated: 2026-09-07
status: abgeschlossen
tags: [datenprovenienz, generator-lineage, v36-vs-v41, dgp-mechanics, reproduzierbarkeit]
---

# Datenprovenienz & Generator-Zuordnung: V3.1, V3.6 Clean und V4.1

## 1. Ausgangslage & Zielsetzung

In simulationsbasierten Forschungs- und Benchmarkprojekten ist die lückenlose Nachvollziehbarkeit (**Data Provenance / Lineage**) der erzeugten Datensätze von zentraler Bedeutung. Da der Data Generating Process (DGP) im Projektverlauf von Version 3.1 über V3.6 bis zu V4.1 kontinuierlich verfeinert wurde, muss für jeden Datensatz exakt feststehen:
1. **Welches Generatorskript** zu welchem Zeitpunkt und Commit-Stand die CSV-Dateien geschrieben hat.
2. **Welche Parameter und Seeds** (Population-Seed, Exam-RNG-Synchronisation, Dosis-Multiplikatoren) aktiv waren.
3. **Auf welchen Datensatz sich historische Analysebefunde beziehen** (insbesondere das Rätsel um die „1.064 durch fachlichen Support geschädigten Studierenden“).
4. **Wie künftige Läufe vollautomatisch mit einem standardisierten Metadaten-Header (`generation_metadata.json`) versehen werden.**

---

## 2. Systematische Provenienz- und Generator-Matrix

Die folgende Tabelle ordnet die drei zentralen Datensätze ihren Erzeugungsskripten, Git-Commits und DGP-Konfigurationen zu:

| Dimension / Merkmal | V3.1 / V3.2 (`output_dl_v2`) | V3.6 Clean (`src/output_dl_v36_clean`) | V4.1 Baseline S01 (`data_v4_grid/S01_baseline`) |
| :--- | :--- | :--- | :--- |
| **Generatorskript** | `legacy_code/archive_scripts/simulation_v2.py` | `legacy_code/archive_scripts/simulation_v3.py` | `legacy_code/archive_scripts/run_v4_simulation_grid.py` + `simulation_v4.py` (jetzt modular in `deepsupport.simulation`) |
| **Git-Commit (Erzeugung)** | Frühe Phase (ca. 11.08.2026) | `360477c` / `d091c77` (26.08.2026) | `0288f2e` / `f8fca84` (30.08.2026) |
| **Population Seed** | `42` / `12345` | `12345` (in `main()`), `42` in `config` | `99999` |
| **Stichprobengröße ($N$)** | $50.000$ | $50.000$ | $50.000$ (pro Universum, 15 Szenarien) |
| **Universen-Konfiguration** | 5 Welten: A, B, C, D, E | 5 Welten: A, B, C, D, E | 8 Welten: A, B, C, D, E, F, G, H |
| **Dropout-Rate Universum A** | $25.99\%$ ($12.993$ Abbrüche) | $25.29\%$ ($12.645$ Abbrüche) | $29.21\%$ ($14.607$ Abbrüche) |
| **Dropout-Rate Universum B** | $29.79\%$ ($14.893$ Abbrüche) | $28.32\%$ ($14.160$ Abbrüche) | $37.11\%$ ($18.554$ Abbrüche) |
| **ARR (B vs. A)** | $3.80\text{ pp}$ | $3.03\text{ pp}$ | $7.90\text{ pp}$ |
| **`support_effect_multiplier`** | Nicht vorhanden (implizit $1.0\times$) | Nicht vorhanden (implizit $1.0\times$) | **$5.0\times$** (Baseline S01) |
| **Effektive Motivationsdosis** | $+0.020$ | $+0.020$ | **$+0.100$** |
| **Notenboost-Gewicht** | $0.08$ | $0.08$ | $0.08$ (S04: 0.04, S05: 0.16, S06: 0.32) |
| **Prüfungsrauschen** | Normal(0, 0.18) | Normal(0, 0.18), positionsdeterministisch | Normal(0, 0.18), universen-synchronisiert |
| **Overload- & Abwurf-Logik** | Strikt deterministisch: Overload $>150\text{h} \implies$ Abwurf härtestes Modul; Penalty harter Cap bei 0.15 | Stochastischer Puffer: `rng_support.random() < 0.2` fängt akute Überlastung ab | Stetige Sigmoid-Wahrscheinlichkeit für Abwurf; stetige Overload-Penalty |
| **Apathy-Dampening** | Nein | Nein | Ja (Motivation $< 0.3 \implies$ gedämpfte Überfachlich-Quote) |

---

## 3. Die historische Auflösung der 1.064 „geschädigten Studierenden“

In früheren Notizen und Diskussionsprotokollen (vgl. `00_Historisches_Gesamtprotokoll.md`, Prompts #35–#38) tauchte der Befund auf, dass durch die Einführung von fachlichem Support **1.064 Studierende in den Dropout getrieben wurden**, während nur **1.340 gerettet** wurden (Netto-Effekt: magere $+276$ Absolventen).

Ein direkter mikro-empirischer Abgleich der Ergebnis-Dateien (`abschluesse.csv` zwischen Universum A [Full Support] und Universum C [Kein fachlicher Support]) klärt diesen Sachverhalt vollständig auf:

### Empirischer 3-Generationen-Abgleich (Universum A vs. Universum C, $N = 50.000$)

| Kohorte / DGP-Stand | Gerettet durch fachl. Support (A: Abschluss, C: Dropout) | Geschädigt durch fachl. Support (A: Dropout, C: Abschluss) | Netto-Gewinn durch fachl. Support | Interpretation & Ursache |
| :--- | :---: | :---: | :---: | :--- |
| **V3.1 / V3.2 (`output_dl_v2`)** | $1.340$ ($2.68\%$) | **$1.064$ ($2.13\%$)** | **$+276$ ($+0.55\%$)** | **Der historische Ursprung:** Harter Modulabwurf bei $+30\text{h}$ Workload. Die Zeitkosten fraßen den Notenbonus fast vollständig auf. |
| **V3.6 Clean (`src/output_dl_v36_clean`)** | $1.179$ ($2.36\%$) | **$67$ ($0.13\%$)** | **$+1.112$ ($+2.22\%$)** | **Entschärfung:** Der in V3.6 integrierte stochastische Zeitpuffer (`rng < 0.2`) reduzierte die Overload-Opfer um $93.7\%$! |
| **V4.1 Baseline S01 (`data_v4_grid`)** | $1.672$ ($3.34\%$) | **$207$ ($0.41\%$)** | **$+1.465$ ($+2.93\%$)** | **V4.1-Realismus:** Kontinuierliche Overload-Penalty und Sigmoid-Abwurf. Hohe Schutzwirkung bei minimaler Verdrängung. |

> [!IMPORTANT]
> **Fazit zur 1.064-Zahl:**
> Die Zahl von $1.064$ Opfern war ein realer, reproduzierbarer Befund der **V3.1/V3.2-Simulation**, der damals richtigerweise zur Überarbeitung der Abwurfmechanik führte. In **V3.6 Clean** existierte dieses Phänomen bereits fast nicht mehr ($67$ Fälle), und in **V4.1** überwiegt der Nutzen den Schaden um das Achtfache ($1.672$ vs. $207$).

---

## 4. Die Chronologie des `support_effect_multiplier`

Die Frage, warum beobachtende Kausalmodelle (wie DML) in V3.6 kaum Schutzwirkung sahen ($HR \approx 0.98$), in V4.1 aber exakt die Ground Truth ($HR \approx 0.85$) isolieren, hängt eng mit der Kalibrierungsgeschichte des Multiplikators zusammen:

1. **V3.6 Clean:**
   - Parameter `support_effect_multiplier` existierte nicht in `config.py`.
   - Jeder überfachliche Support addierte pauschal $+0.02$ zur Motivation.
2. **Erste V4-Grid-Suche (dokumentiert in `diagnose_gridsearch_v4.md`):**
   - In `config.py` war die Baseline versehentlich auf `support_effect_multiplier = 5.0` vordefiniert.
   - Die Szenarien S02 und S03 übergaben relative Werte: S02 übergab `0.5` und S03 übergab `2.0`.
   - Weil `5.0` der Standard war, wirkte `0.5` als $0.1\times$ der Baseline (eine $80\%$-Dosisreduktion!) und `2.0` als $0.4\times$ der Baseline. Dies führte zum scheinbar paradoxen Befund, dass S03 schlechter schützte als S01.
3. **V4.1 Bereinigte Grid-Suche (Nachtlauf 30.08.2026):**
   - Baseline S01: Explizit auf `support_effect_multiplier = 5.0` (Dosis $+0.10$).
   - S02 (`supp_half`): Auf `2.5` kalibriert ($0.5\times$ der Baseline, Dosis $+0.05$).
   - S03 (`supp_double`): Auf `10.0` kalibriert ($2.0\times$ der Baseline, Dosis $+0.20$).

---

## 5. Die Signal-to-Confounder Ratio (SCR) im Detail

Die quantitative Erklärung für das Phänomen „Faktor 14“ stützt sich auf die Relation zwischen Kausaldosis und Selektionsbias:

$$\text{SCR} = \frac{\text{Effektive Behandlungsdosis}}{\text{Selektionsbias in latenter Motivation } |\Delta \text{Mot}|}$$

* **In V3.6 Clean:**
  * Mittlere Motivation Nicht-Nutzer: $0.7168$
  * Mittlere Motivation Support-Nutzer: $0.4453$
  * Selektionsgraben: $\Delta = -0.2715$ (Demotivierte wählen massiv Support)
  * Dosis: $+0.0200$
  * $\text{SCR}_{\text{V3.6}} = \frac{0.0200}{0.2715} = 7.37\%$
* **In V4.1 Baseline S01:**
  * Mittlere Motivation Nicht-Nutzer: $0.6559$
  * Mittlere Motivation Support-Nutzer: $0.5612$
  * Selektionsgraben: $\Delta = -0.0947$ (Apathy-Dampening verhindert extrem demotivierte Teilnahme)
  * Dosis: $+0.1000$
  * $\text{SCR}_{\text{V4.1}} = \frac{0.1000}{0.0947} = 105.60\%$
* **Verhältnis der Signalstärken:**
  $$\frac{\text{SCR}_{\text{V4.1}}}{\text{SCR}_{\text{V3.6}}} = \frac{105.60\%}{7.37\%} = \mathbf{14.33}$$

In V3.6 macht das Kausalsignal weniger als $8\%$ des Selektionsbias aus. In V4.1 übertrifft das Kausalsignal den Selektionsbias ($105\%$). Dadurch kann ein Deep-Learning-Verfahren wie DML in V4.1 das Signal mühelos von Confounding trennen, während es in V3.6 unter die Identifizierbarkeitsgrenze fällt.

---

## 6. Automatische Datensatz-Provenienz (`generation_metadata.json`)

Um künftig jegliche Ambiguität über Datensatzursprünge auszuschließen, wurde in `src/export.py` die Funktion `schreibe_generation_metadata()` integriert. Jeder Aufruf von `exportiere_csv()` erzeugt nun automatisch eine Datei `generation_metadata.json` im Zielverzeichnis.

### Schema der Metadaten (`schema_version: "1.0"`):

```json
{
  "schema_version": "1.0",
  "timestamp_utc": "2026-09-07T21:12:37.722782+00:00",
  "git": {
    "commit": "b5374db17bb9779dcb795f291b75df2f29958a9b",
    "branch": "main",
    "dirty": false
  },
  "environment": {
    "python_version": "3.12.x ...",
    "platform": "Windows-11-...",
    "executable": "C:\\GitHub_public\\.venv\\Scripts\\python.exe"
  },
  "generator": {
    "entry_script": "src/run_v4_universes.py",
    "generator_script": "c:/GitHub_public/Abschlussprojekt/src/run_v4_universes.py"
  },
  "config": {
    "seed_population": 99999,
    "n_studierende": 50000,
    "support_effect_multiplier": 5.0,
    "gewicht_support_boost": 0.08,
    "gewicht_rauschen": 0.18,
    "support_kosten_faktor": 1.0,
    "overload_penalty_factor": 0.1
  },
  "extra_info": {
    "scenario_id": "S01_baseline",
    "universe": "A",
    "universe_label": "Alle Support-Typen erlaubt"
  }
}
```

Dadurch ist jeder Datenordner in sich abgeschlossen und ohne externe Annahmen bis auf den exakten Git-Commit auditierbar.

---

## Verwandte Dokumente

| Dokument | Pfad | Relevanz für diesen Bericht |
| :--- | :--- | :--- |
| **Master-Synopse V4 Gesamt** | [master_synopse_v4_gesamt.md](../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md) | Gesamtevaluation aller Modelle über V4.1-Szenarien |
| **Sensitivitätsanalyse V4.1 Nachtlauf** | [sensitivitaetsanalyse_v41_nachtlauf.md](../04_causal_and_simulation/sensitivitaetsanalyse_v41_nachtlauf.md) | Detaillierte Auswertung der 15 V4.1-Szenarien |
| **Support-Effekte Analyse** | [support_effects_analysis.md](../04_causal_and_simulation/support_effects_analysis.md) | Historische Analyse der Universen A bis E |
| **Diagnose Gridsearch V4** | [diagnose_gridsearch_v4.md](../06_misc/diagnose_gridsearch_v4.md) | Ursprüngliche Diagnose der Multiplikator-Inversion (S02/S03) |
| **Historisches Gesamtprotokoll** | [00_Historisches_Gesamtprotokoll.md](../07_conversation_logs/00_Historisches_Gesamtprotokoll.md) | Chronologische Entwicklung und User-Prompts zu den 1.064 Fällen |
