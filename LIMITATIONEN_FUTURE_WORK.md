# Limitationen & Future Work

Dieses Dokument fasst die methodischen, datentechnischen und konzeptionellen Einschränkungen des Projekts zusammen und skizziert Ansätze für zukünftige Arbeiten.

---

## 1. Limitationen der Simulations-Engine (`simulation.py`)

### A. Studiengangs-Modellierung & Fachbereichsklima
Der synthetische Generator unterscheidet Studiengänge derzeit primär über:
- Die Abfolge, Anzahl und Credit-Points der Pflichtmodule
- Die Modul-Schwierigkeitsgrade und Prüfungshürden
- Die empirische Geschlechterverteilung pro Studiengang

**Einschränkung:** Es gibt derzeit **keine domänenspezifische Modellierung von Fachbereichskulturen**, sozialen Stereotype-Threats (z. B. MINT-spezifisches Hürdenklima für Frauen nach DZHW-Studien) oder psychologischen Gruppenprofilen. Alle Studierenden teilen dieselbe fundamentale Motivations- und Integrationsdynamik.

*Future Work:* Erweiterung des Generators um fachbereichsspezifische Klimafaktoren und empirisch kalibrierte Motivationsverläufe nach DZHW-/CHE-Studierendensurveys.

### B. Parametrische Annahmen
Die Parameter der Simulation (z. B. `gewicht_support_boost = 0.04`, `p += 0.20` nach Fehlversuch) wurden heuristisch-plausibel gewählt. Obwohl eine Ground Truth existiert, wurde bisher keine systematische Parameter-Sensitivitätsanalyse (Grid Search über Generator-Seeds) durchgeführt.

---

## 2. Methodische Limitationen der Kausalinferenz

### A. Reaktiver Confounding Bias & Extrapolation in Sequenzmodellen
Im Datengenerator treten Studierende dann dem Support bei, wenn sie sich in einer **akuten Krise** befinden (z. B. verhauene Prüfung $\rightarrow +20\%$ Support-Chance). 

**Einschränkung:** Rekurrierende Modelle (GRU, DeepHit) lernen über ihre temporale Historie, dass das Signal `support_active = 1` ein Indikator für ein tiefes Leistungs- und Motivationstief ist. Werden im kontrafaktischen Testset alle Support-Flags auf `1` gesetzt, extrapolieren Sequenzmodelle außerhalb ihrer Trainingsverteilung, da sie "Dauer-Support in Krisen" selten gesehen haben.

*Future Work / Lösung:* Weiterentwicklung des implementierten **Double Machine Learning (DML)** Ansatzes (`dml_orthogonal_survival.py`) mit zweistufiger Residual-Orthogonalisierung (2SRI).

### B. KPI-Operationalisierung im Dashboard
Die in `kpi.md` definierten Metriken für Wirksamkeit und Zielgruppenerreichung wurden im Analyse-Backend berechnet und geloggt, aus Zeitgründen jedoch nicht vollständig in der Benutzeroberfläche des Dashboards visualisiert.

---

## 3. Dokumentations- & Artefakt-Transparenz

Sämtliche Analyseberichte, Modellvergleiche und Audit-Protokolle im Ordner `Artifacts/` sowie in der System-Historie wurden in transparenter Paarprogrammierung und direkter Generierung durch KI-Agenten (Antigravity IDE, Claude Opus/Sonnet, Gemini Pro/Flash) erstellt.
