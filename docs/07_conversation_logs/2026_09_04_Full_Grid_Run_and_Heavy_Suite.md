---
## 📝 Entwicklungs-Historie & User-Annotation
**Datum:** 04. September 2026  
**Kontext:** Abschluss des V4.2 Master Grid Runs (225 Modelle) und der Heavy Deep Suite (Homeserver ThinkCentre M70q & Workstation)  
**Auslösende Annotationen (Zusammenfassung):**
> *"Also ich habe gestern Nacht die heavy suite auf dem LXC gestartet. Leider habe ich inzwischen die Verbindung zum homeserver verloren, vermutlich, weil die Intel NIC ausgefallen ist..."*
> *"So, habe ich gemacht, ist das dauerhaft? Das sind die Ergebnisse des Laufs auf dem Heimservers. Ich werde sie gleich mal puschen."*
> *"Mit der NIC hat alles geklappt, denke ich. Ich hoffe nur, dass es auch den Fehler behebt. Danke Dir für die Übersicht über die Werte des Laufs der Heavy Suite, man sieht, das Rauschen ist ein extrem starker Einflußfaktor..."*
> *"Ok, kannst Du bitte nochmal die Dokumentationsroutine durchführen..."*
---

# Protokoll: V4.2 Grid Run Abschluss, Homeserver Cluster Execution & Heavy Suite Synopse

## 1. Ausgangslage & Herausforderung
Nach dem erfolgreichen Refactoring des `deepsupport/` Modulpakets und der I/O-Absicherung des `grid_runner.py` standen zwei komplementäre Rechenlasten an:
1. **Light Feature Grid:** Vollständiger 15-Szenarien-Lauf (S01–S15) × 3 Modellarchitekturen (`grid_semester_gru`, `grid_semester_transformer`, `grid_exam_gru`) × 5 Feature-Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`) = **225 DL-Modelle** auf $N=50.000$ Studierenden.
2. **Heavy Deep Suite:** Autoregressive Next-Exam-Vorhersage (Dual-Head GRU), Fail-Focus PR-AUC Evaluation, Deep Autoregressive Transformer mit $\sin/\cos$ Positional Encoding und Landmark Representation Learning (Ende Sem 2).

Um die Workstation nicht durch beide Mammutläufe tagelang zu blockieren, wurde eine **heterogene Cluster-Strategie** gewählt: Der Master Grid Run lief auf der Workstation (HP EliteDesk), während die Heavy Suite auf den Lenovo ThinkCentre M70q Homeserver (Debian LXC unter Proxmox VE) ausgelagert wurde.

---

## 2. Der Homeserver Intel-NIC-Incident & Hardware-Fix
Während des nächtlichen Heavy-Suite-Laufs verlor der Homeserver seine Netzwerkverbindung. 

### Diagnose
- **Hardware:** Lenovo ThinkCentre M70q mit Intel I219-LM/V Gigabit Ethernet Controller (`e1000e` Treiber) unter Linux Kernel 6.x / Proxmox VE.
- **Root Cause:** Bekannter Treiber-Bug unter anhaltender I/O- und CPU-Last: *Energy Efficient Ethernet (EEE)* und *TCP Segmentation Offloading (TSO/GSO)* lösen einen Hardware-Unit-Hang aus, der die Schnittstelle stummschaltet.
- **Schnittstellen-Zuordnung:** In Proxmox ist die physische Karte als **`nic0`** (mit `altname enp0s31f6`) an die Linux-Bridge `vmbr0` gebunden. Im LXC-Container `PythonLXC` existiert nur die virtuelle `eth0`.

### Lösung & Persistierung
1. Sofortiges Deaktivieren von EEE und Offloading auf dem Proxmox-Host:
   ```bash
   ethtool --set-eee nic0 eee off
   ethtool -K nic0 tso off gso off
   ```
2. Dauerhafte Persistierung in `/etc/network/interfaces` des Proxmox-Hosts:
   ```text
   iface nic0 inet manual
       post-up ethtool --set-eee nic0 eee off
       post-up ethtool -K nic0 tso off gso off
   ```
Der Homeserver überstand den Lauf und der User konnte die generierten Keras-Modelle und Metriken per Commit `a1c0f69` nach GitHub pushen.

---

## 3. Code-Bugfixes & Lokale Re-Evaluation
Aufgrund geringfügiger Versionsunterschiede brachen Step 2 und Step 4 auf dem Homeserver ab:
- **Step 2 (`eval_autoregressive_fail.py`):** Fehlender Import von `from sklearn.preprocessing import StandardScaler` behoben.
- **Step 4 (`landmark_prediction.py`):** Scikit-Learn `HistGradientBoostingClassifier` wirft bei `n_estimators` einen `TypeError`; robuster Fallback auf `max_iter=100` implementiert.

Beide Teilschritte wurden auf der Workstation für alle drei Rausch-Szenarien (`S01_baseline`, `S07_noise_half`, `S08_noise_double`) fehlerfrei re-evaluiert und persistiert.

---

## 4. Wichtigste empirische Erkenntnisse

### A. Transformer deklassiert GRU auf Exam-Level
- In allen drei Rauschniveaus übertrifft der **Deep Transformer mit Sinusoidal Positional Encoding** das **Dual-Head GRU** deutlich bei der kontinuierlichen Notenvorhersage:
  - S07 (Halbes Rauschen): $R^2 = \mathbf{0.8623}$ vs. $0.6135$ (**$+0.25$ Punkte**)
  - S01 (Baseline): $R^2 = \mathbf{0.6996}$ vs. $0.5659$ (**$+0.13$ Punkte**)
  - S08 (Doppeltes Rauschen): $R^2 = \mathbf{0.3825}$ vs. $0.3051$ (**$+0.08$ Punkte**)
- *Erklärung:* Multi-Head Self-Attention umgeht den informationellen Flaschenhals des sequentiellen Hidden-States und kann direkt auf inhaltliche Vorläufermodule (z. B. Mathe I für Statistik II) fokussieren.

### B. Frühwarnsysteme und Fail PR-AUC (Step 2)
- Das Nicht-Bestehen (Fail) tritt mit ~14–20 % Prävalenz auf.
- In S07 erreicht die Fail-Vorhersage eine PR-AUC von **0.6573** (ein **4.5-facher Precision-Lift** über Zufallsniveau). In der Baseline liegt die PR-AUC bei **0.3801** (2.3-facher Lift).

### C. Landmark Representation Learning (Step 4)
- Nach nur **2 Fachsemestern** erklären die latenten Transformer-Embeddings bereits **76.5 % der Varianz der finalen Studienabschlussnote** (S01; bei S07: **86.8 %**).
- Künftige Absolventen werden mit einem **F1-Score von 0.90** diskriminiert. Das 1. Studienjahr ist der entscheidende Weichensteller.

---

## 5. Dokumentations- & Artefakt-Status
- [`docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md): Vollständiger Synopsen-Bericht zur Heavy Suite.
- [`docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md): Master-Synopse über alle 15 Szenarien und 225 Modelle.
- [`docs/06_misc/system_and_hardware_stack.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/06_misc/system_and_hardware_stack.md): Dokumentation der Cluster-Knoten und Hardware-Konfiguration.
