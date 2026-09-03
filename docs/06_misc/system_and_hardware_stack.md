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

---

## 2. Homeserver & Rechen-Node (Virtualisierungs-Host)

* **Modell:** Lenovo ThinkCentre M70q Tiny
* **Prozessor (CPU):** Intel Core i5-10400T (6 Kerne / 12 Threads, 2.00 GHz Basis / bis zu 3.60 GHz Turbo, Low-Power 35W TDP)
* **Arbeitsspeicher (RAM):** 32 GB DDR4 RAM
* **Hypervisor / OS:** Proxmox VE (Virtual Environment)
* **Rechenumgebung:** 
  * **LXC-Container (Debian Linux):** Schlanker, ressourcensparender Container ohne Virtualisierungs-Overhead
  * **Einsatzzweck:** Autarke Ausführung von Batch-Skripten, rechenintensiven Hintergrund-Läufen und geplanter Ziel-Host für schlanke PyTorch/PyCox-Microservices

---

## 3. Architektur-Implikationen für das Modell-Design

| Kriterium | HP EliteDesk G5 (Windows 11) | Lenovo ThinkCentre M70q (LXC Debian) |
| :--- | :--- | :--- |
| **Rechen-Fokus** | Interaktive Analyse, Feature-Engineering, EDA, Antigravity Agent Sessions | Headless Batch-Runs, Cron-Jobs, nächtliche Grid-Berechnungen |
| **I/O-Profil** | Extrem hohe Random-Read/Write-Performance dank Lexar NM790 (ideal für DuckDB In-Memory/Disk-Spill) | Kontinuierliche Dauerlast bei minimaler Leistungsaufnahme (35W TDP) |
| **Framework-Eignung** | Keras/TensorFlow (mit gepinnten Windows Wheels), Scikit-Learn | **PyTorch 2.x & PyCox / 1D-TCN:** Schlankes C++ Backend, minimale RAM-Belegung, perfekte CPU-Vektorisierung im Linux-Kernel |
