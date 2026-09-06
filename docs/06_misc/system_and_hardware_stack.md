---
created: 2026-09-04
last_updated: 2026-09-06
status: abgeschlossen
tags: [infrastruktur, hardware, system, venv, cluster]
---

# System- & Hardware-Stack (Infrastruktur)

Dieses Dokument erfasst die genauen Spezifikationen der genutzten Entwicklungs-, Trainings- und Server-Infrastruktur für das Projekt **DeepHSSurvival**.

---

## 1. Primärer Arbeitsrechner (Workstation / Client)

* **Modell:** HP EliteDesk 800 G5 Desktop Mini (Performance Edition)
* **Prozessor (CPU):** Intel Core i5-9500 (6 Kerne / 6 Threads, bis zu 4.40 GHz Turbo, 9 MB Cache)
* **Arbeitsspeicher (RAM):** 32 GB DDR4 RAM
* **Massenspeicher (SSD):** Lexar NM790 M.2 PCIe 4.0 NVMe SSD (High-Speed I/O für DuckDB- und Parquet-Transaktionen)
* **Betriebssystem:** Microsoft Windows 11 Pro (64-Bit)
* **Primäre Software & IDE:**
  * **AI-Pair-Programming & Agentic Workspace:** Google Antigravity (Advanced Agentic Coding Environment)
  * **Shell:** PowerShell 7 / Windows Terminal
  * **Python-Stack:** Python 3.12 (Virtualenv `C:\GitHub_public\.venv`), TensorFlow / Keras 3, DuckDB, NumPy, Pandas, Scikit-Survival, Lifelines

### ⚠️ Wichtige Sicherheits- & Ausführungs-Richtlinie: Windows Application Control
Unter Windows 11 ist auf dem Host eine strikte *Application Control Policy* (AppLocker / Windows Defender Application Control) aktiv:
- **System-Python:** Das Standard-Python unter `C:\Users\wilfr\AppData\Local\Programs\Python\...` blockiert beim Laden nativer Binaries (z. B. SciPy HiGHS Solver DLLs `scipy.optimize._highspy._core`, C++ Extensions).
- **Whitelisted Virtual Environment:** Das dedizierte Virtual Environment unter `C:\GitHub_public\.venv` ist vollständig für native Binaries whitelisted und installationsberechtigt.
- **Vorgeschriebenes Ausführungsmuster:**
  ```powershell
  $env:PYTHONPATH = "src"
  C:\GitHub_public\.venv\Scripts\python.exe <script_pfad> [argumente]
  ```

---

## 2. Homeserver & Rechen-Node (Virtualisierungs-Host)

* **Modell:** Lenovo ThinkCentre M70q Tiny
* **Prozessor (CPU):** Intel Core i5-10400T (6 Kerne / 12 Threads, 2.00 GHz Basis / bis zu 3.60 GHz Turbo, Low-Power 35W TDP)
* **Arbeitsspeicher (RAM):** 32 GB DDR4 RAM
* **Hypervisor / OS:** Proxmox VE (Virtual Environment)
* **Rechenumgebung:** 
  * **LXC-Container (Debian Linux):** Schlanker, ressourcensparender Container ohne Virtualisierungs-Overhead
  * **Einsatzzweck:** Autarke Ausführung von Batch-Skripten, rechenintensiven Hintergrund-Läufen (z. B. Heavy Suite Grid) und geplanter Ziel-Host für schlanke PyTorch/PyCox-Microservices
* **Netzwerk-Konfiguration (Intel I219-LM NIC Fix):**
  * EEE (Energy Efficient Ethernet) und TCP Offloading (TSO/GSO) wurden zur Vermeidung von Treiber-Hangs unter Dauerlast dauerhaft deaktiviert (`ethtool --set-eee nic0 eee off`, `ethtool -K nic0 tso off gso off`).

---

## 3. Architektur-Implikationen für das Modell-Design

| Kriterium | HP EliteDesk G5 (Windows 11) | Lenovo ThinkCentre M70q (LXC Debian) |
| :--- | :--- | :--- |
| **Rechen-Fokus** | Interaktive Analyse, Feature-Engineering, EDA, Antigravity Agent Sessions | Headless Batch-Runs, Cron-Jobs, nächtliche Grid-Berechnungen |
| **I/O-Profil** | Extrem hohe Random-Read/Write-Performance dank Lexar NM790 (ideal für DuckDB In-Memory/Disk-Spill) | Kontinuierliche Dauerlast bei minimaler Leistungsaufnahme (35W TDP) |
| **Framework-Eignung** | Keras/TensorFlow (mit gepinnten Windows Wheels), Scikit-Learn | **PyTorch 2.x & PyCox / 1D-TCN:** Schlankes C++ Backend, minimale RAM-Belegung, perfekte CPU-Vektorisierung im Linux-Kernel |

---

## Verwandte Dokumente

| Dokument | Bezug |
|:---|:---|
| [AGENTS.md](../../AGENTS.md) | Verbindliche Ausführungsregeln für Coding-Agenten |
| [Heavy Suite Synopse](../03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md) | Rechenergebnisse des Homeserver-Laufs |
| [Protokoll Homeserver Incident](../07_conversation_logs/2026_09_04_Full_Grid_Run_and_Heavy_Suite.md) | Technische Details zum Intel NIC Hang und Fix |
| [DeepSupport_Projektentwicklung](../08_project_evolution/DeepSupport_Projektentwicklung.md) | Hardware- und Infrastrukturentwicklung im Gesamtkontext |
