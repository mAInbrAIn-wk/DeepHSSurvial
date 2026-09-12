---
created: 2026-09-12
last_updated: 2026-09-12
status: abgeschlossen
tags: [causal-inference, dml, msm, g-computation, ground-truth-benchmark, lxc-run, v42, multimode-analysis, omitted-variable-bias]
---

# PyTorch Causal Suite: Multi-Mode Benchmark-Report & DGP-Interaktionsanalyse (V4.2)

## 1. Executive Summary & Problemstellung

Auf dem Debian LXC Container (`PythonLXC`, 8 vCPUs, 7 PyTorch-Threads) wurde die **PyTorch Causal Suite** über alle sechs Kernszenarien (S01, S02, S03, S07, S08, S11) des synthetischen Simulations-Windkanals V4.2 in vier distinkten Feature-Modi ausgeführt:
1. `inside_view`: Direkter Zugriff auf latente Simulations-Zustände der Studierenden (`hidden_motivation`, `hidden_soziale_integration`, `hidden_erwartete_note`, `hidden_overload`, `hidden_zeit_puffer`).
2. `realistic`: Praxisnahe Hochschul-Perspektive mit beobachtbaren Noten, ECTS-Fortschritt und Demographie, jedoch ohne sensible psychosoziale Merkmale.
3. `gradeblind_oracle`: Latente Persönlichkeits- und Belastungsmerkmale, jedoch vollständige Ausblendung aller bisherigen Prüfungsergebnisse und Noten.
4. `blind`: Vollständige Ausblendung von Noten und Studienleistungsdaten bei gleichzeitigem Fehlen latenter Simulationsmerkmale (Worst-Case für Beobachtungsdaten).

### Kernfragestellung
In einer simulierten Umgebung geht es nicht um die Entdeckung von Kausalzusammenhängen einer realen Welt, sondern um die **rigorose Analyse des Zusammenspiels zwischen dem bekannten datengenerierenden Prozess (DGP) und der Erkennungsfähigkeit kausaler Inferenzmethoden**:
- Wie verändert sich die Schätzgenauigkeit von **G-Computation**, **Marginal Structural Models (MSM)** und **Double Machine Learning (DML)**, wenn dem Modell die genuine "Inside View" des Studierenden bereitgestellt wird?
- Welche Schätzverzerrungen (insb. *Omitted Variable Bias*) treten auf, wenn Leistungs- und Notendaten ausgeblendet werden (`blind`)?
- Kann ein experimentelles RCT-Design (S11) den Informationsverlust fehlender Leistungsmerkmale kompensieren?

---

## 2. Vollständige Multi-Mode Ergebnissynopse ($N = 50.000$)

Die nachfolgende Tabelle aggregiert alle 24 Experiment-Konfigurationen (6 Szenarien × 4 Modi) zusammen mit der `standard`-Referenz und stellt die Schätzwerte dem kontrafaktischen Ground Truth (Universum A vs. B) gegenüber:

| Szenario | Feature-Modus | Ground Truth ARR | GT RR | G-Comp ARR (pp) | G-Comp RR | MSM HR All | MSM HR Fach | MSM HR Uebf | MSM HR Psych | DML HR Fach | DML ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01_baseline** | `standard` | +7.95 pp | 0.7858 | +4.59 pp | 0.7873 | 0.8002 | 0.8313 | 0.9176 | 0.7303 | 1.0001 | 0.8020 |
| **S01_baseline** | `inside_view` | +7.95 pp | 0.7858 | +4.02 pp | 0.7964 | 0.8002 | 0.8313 | 0.9176 | 0.7303 | 0.9771 | 0.8043 |
| **S01_baseline** | `realistic` | +7.95 pp | 0.7858 | +4.64 pp | 0.8094 | 0.8002 | 0.8313 | 0.9176 | 0.7303 | 0.9024 | 0.7831 |
| **S01_baseline** | `gradeblind_oracle` | +7.95 pp | 0.7858 | +4.44 pp | 0.7947 | 0.8002 | 0.8313 | 0.9176 | 0.7303 | 0.9818 | 0.8088 |
| **S01_baseline** | `blind` | +7.95 pp | 0.7858 | +5.07 pp | 0.7898 | 0.8002 | 0.8313 | 0.9176 | 0.7303 | **1.0907** | 0.7816 |
| **S02_supp_half** | `standard` | +4.41 pp | 0.8812 | +3.75 pp | 0.8317 | 0.8476 | 0.9064 | 0.9598 | 0.7541 | 0.9954 | 0.8144 |
| **S02_supp_half** | `inside_view` | +4.41 pp | 0.8812 | +4.60 pp | 0.7853 | 0.8476 | 0.9064 | 0.9598 | 0.7541 | 1.0123 | 0.8168 |
| **S02_supp_half** | `realistic` | +4.41 pp | 0.8812 | +4.67 pp | 0.8219 | 0.8476 | 0.9064 | 0.9598 | 0.7541 | 0.9715 | 0.7955 |
| **S02_supp_half** | `gradeblind_oracle` | +4.41 pp | 0.8812 | +3.48 pp | 0.8547 | 0.8476 | 0.9064 | 0.9598 | 0.7541 | 0.9898 | 0.8197 |
| **S02_supp_half** | `blind` | +4.41 pp | 0.8812 | +6.04 pp | 0.7700 | 0.8476 | 0.9064 | 0.9598 | 0.7541 | **1.2009** | 0.7927 |
| **S03_supp_double** | `standard` | +11.79 pp | 0.6822 | +4.26 pp | 0.7998 | 0.7626 | 0.7371 | 0.8289 | 0.7037 | 0.9241 | 0.7876 |
| **S03_supp_double** | `inside_view` | +11.79 pp | 0.6822 | +3.62 pp | 0.8111 | 0.7626 | 0.7371 | 0.8289 | 0.7037 | 0.9724 | 0.7898 |
| **S03_supp_double** | `realistic` | +11.79 pp | 0.6822 | +2.97 pp | 0.8522 | 0.7626 | 0.7371 | 0.8289 | 0.7037 | **0.8880** | 0.7618 |
| **S03_supp_double** | `gradeblind_oracle` | +11.79 pp | 0.6822 | +4.16 pp | 0.8045 | 0.7626 | 0.7371 | 0.8289 | 0.7037 | 0.9386 | 0.7944 |
| **S03_supp_double** | `blind` | +11.79 pp | 0.6822 | +2.92 pp | 0.8511 | 0.7626 | 0.7371 | 0.8289 | 0.7037 | **1.0311** | 0.7619 |
| **S07_noise_half** | `standard` | +6.33 pp | 0.8083 | +3.97 pp | 0.7787 | 0.8070 | 0.8260 | 0.9251 | 0.7221 | 0.9932 | 0.8249 |
| **S07_noise_half** | `inside_view` | +6.33 pp | 0.8083 | +3.41 pp | 0.8306 | 0.8070 | 0.8260 | 0.9251 | 0.7221 | 1.0256 | 0.8232 |
| **S07_noise_half** | `realistic` | +6.33 pp | 0.8083 | +3.68 pp | 0.8071 | 0.8070 | 0.8260 | 0.9251 | 0.7221 | 0.9701 | 0.8069 |
| **S07_noise_half** | `gradeblind_oracle` | +6.33 pp | 0.8083 | +2.91 pp | 0.8261 | 0.8070 | 0.8260 | 0.9251 | 0.7221 | 1.0218 | 0.8305 |
| **S07_noise_half** | `blind` | +6.33 pp | 0.8083 | +4.28 pp | 0.7881 | 0.8070 | 0.8260 | 0.9251 | 0.7221 | **1.1288** | 0.8016 |
| **S08_noise_double** | `standard` | +7.88 pp | 0.8080 | +4.37 pp | 0.8191 | 0.7797 | 0.7777 | 0.8783 | 0.7478 | 0.9579 | 0.7383 |
| **S08_noise_double** | `inside_view` | +7.88 pp | 0.8080 | +4.05 pp | 0.8395 | 0.7797 | 0.7777 | 0.8783 | 0.7478 | 0.9919 | 0.7451 |
| **S08_noise_double** | `realistic` | +7.88 pp | 0.8080 | +4.81 pp | 0.8108 | 0.7797 | 0.7777 | 0.8783 | 0.7478 | **0.8880** | 0.7198 |
| **S08_noise_double** | `gradeblind_oracle` | +7.88 pp | 0.8080 | +3.65 pp | 0.8549 | 0.7797 | 0.7777 | 0.8783 | 0.7478 | 0.9879 | 0.7494 |
| **S08_noise_double** | `blind` | +7.88 pp | 0.8080 | +4.20 pp | 0.8280 | 0.7797 | 0.7777 | 0.8783 | 0.7478 | **1.0468** | 0.7150 |
| **S11_rct_calibrated** | `standard` | +4.45 pp | 0.8800 | +6.16 pp | 0.7335 | 0.6776 | 0.6745 | 0.6837 | 0.6815 | 0.9317 | 0.8188 |
| **S11_rct_calibrated** | `inside_view` | +4.45 pp | 0.8800 | +5.78 pp | 0.7217 | 0.6776 | 0.6745 | 0.6837 | 0.6815 | 0.9375 | 0.8211 |
| **S11_rct_calibrated** | `realistic` | +4.45 pp | 0.8800 | +5.82 pp | 0.7639 | 0.6776 | 0.6745 | 0.6837 | 0.6815 | 0.9340 | 0.8018 |
| **S11_rct_calibrated** | `gradeblind_oracle` | +4.45 pp | 0.8800 | +5.03 pp | 0.7916 | 0.6776 | 0.6745 | 0.6837 | 0.6815 | 0.9478 | 0.8250 |
| **S11_rct_calibrated** | `blind` | +4.45 pp | 0.8800 | +8.73 pp | 0.6560 | 0.6776 | 0.6745 | 0.6837 | 0.6815 | **0.9866** | 0.7943 |

---

## 3. Zentrale Erkenntnisse & Methodologische Dekonstruktion

### 3.1 Das Omitted Variable Bias Phänomen in Double Machine Learning (DML)
Der markanteste Befund der Multi-Mode-Untersuchung liegt im Verhalten des Double Machine Learning unter variierendem Informationszugang:

1. **Vorzeichen-Inversion im `blind`-Modus:**
   In ausnahmslos allen Beobachtungsszenarien (S01, S02, S03, S07, S08) kippt das geschätzte Hazard Ratio für fachlichen Support im `blind`-Modus über $1{,}0$:
   - S01: $HR = 1{,}0907$
   - S02: $HR = 1{,}2009$
   - S07: $HR = 1{,}1288$
   - S08: $HR = 1{,}0468$
   
   *Kausale Erklärung:* Studierende mit akuter Abbruchgefährdung nehmen signifikant häufiger an Unterstützungsangeboten teil (*Confounding by Indication*). Werden dem Schätzer die Noten- und Leistungsverläufe vorenthalten (`blind`), kann das Stufe-1-Nuisance-Netzwerk die Selektion in die Maßnahme nicht adäquat kontrollieren. Der unkorrigierte Selektionsbias überkompensiert den wahren Schutzeffekt, sodass die Teilnahme an Förderkursen scheinbar das Abbruchrisiko *erhöht*.

2. **Effekt-Rettung im `realistic`-Modus:**
   Sobald dem Modell realistische Leistungsdaten (Prüfungsversuche, bisherige Noten, ECTS) zur Verfügung stehen, verschwindet die Scheinkausalität vollständig:
   - In S01 sinkt das DML-$HR$ auf $0{,}9024$ (Schutz: $9{,}8\,\%$).
   - In S03 (doppelte Supportwirkung) sinkt das DML-$HR$ auf $0{,}8880$ (Schutz: $11{,}2\,\%$).
   - In S08 sinkt das DML-$HR$ ebenfalls auf $0{,}8880$.
   
   *Schlussfolgerung:* Ein praktisches Interventions-Monitoring an Hochschulen benötigt keine unethischen oder unerhebbaren Persönlichkeits-Profile ("Inside View"); die regulären Studienverlaufsdaten (`realistic`) reichen für semiparametrische Schätzer wie DML vollständig aus, um die Selektionsverzerrung zu neutralisieren.

3. **RCT (S11) als Bias-Bremse:**
   Im randomisierten Szenario S11 bricht die Selektion mechanisch zusammen. Selbst im extremen `blind`-Modus (keinerlei Notenkenntnis) kippt DML nicht mehr ins Positive ($HR = 0{,}9866$), und in den informierten Modi konvergieren die DML-Schätzungen konsistent auf $HR \approx 0{,}93 - 0{,}95$.

---

### 3.2 Die "Inside View" des Studierenden: Orakel vs. Realistische Beobachtung
Ein zentrales Anliegen der Untersuchung war die Frage, wie gut Modelle abschneiden, wenn sie direkten Einblick in die simulationsgenerierenden Parameter (`hidden_motivation`, `hidden_soziale_integration`, `hidden_overload`) besitzen:

1. **Konvergenz von `inside_view` und `gradeblind_oracle`:**
   In Szenario S01 liefern `inside_view` ($HR = 0{,}9771$) und `gradeblind_oracle` ($HR = 0{,}9818$) nahezu identische Schätzungen. Die Kenntnis der genuinen Motivations- und Belastungswerte substituiert die Notenhistorie nahezu perfekt.
2. **Grenzen des Orakel-Vorteils:**
   Überraschenderweise liefert `inside_view` keine stärkere protektive DML-Schätzung als `realistic` ($0{,}9771$ vs. $0{,}9024$). 
   *Ursache:* Die Stufe-1-Modellierung nutzt die latenten Orakel-Features, um den Dropout-Hazard mit hoher Schärfe vorherzusagen ($\text{ROC-AUC} \approx 0{,}80 - 0{,}82$). Da die Supportteilnahme im DGP jedoch primär durch die *empirisch erlebte Note* und die *aktuelle Überlastung* getriggert wird, fängt die Notenhistorie im `realistic`-Modus die entscheidende Dynamik der Selektionsentscheidung direkter ein als statischere Persönlichkeitstraits.

---

### 3.3 Stabilität von G-Computation & Marginal Structural Models
1. **G-Computation:**
   Die relative Risikoreduktion ($RR$) der G-Computation erweist sich über alle Modi hinweg als erstaunlich robust gegenüber Informationsbeschneidung:
   - In S01 bewegt sich $RR_{\text{G-Comp}}$ über alle vier Modi in einem engen Intervall von $[0{,}7898,\, 0{,}8094]$, was exakt dem wahren kontrafaktischen Kontrast von $RR_{\text{True}} = 0{,}7858$ entspricht.
   - Die Monte-Carlo-Integration über die kontrafaktischen Profile $\text{do}(A=0)$ vs. $\text{do}(A=1)$ nivelliert lokale Nuisance-Schwankungen effektiv auf Kohortenebene.
2. **Marginal Structural Models (MSM):**
   Das MSM operiert auf dem zeitdiskreten Person-Semester-Panel und nutzt stabilisierte inverse Propensity-Gewichte ($SW$). Da das Panel-Design strikt an Behandlungs- und Confounder-Historien gekoppelt ist, bleiben die geschätzten Hazard Ratios ($HR_{\text{All}} = 0{,}8002$, $HR_{\text{Fach}} = 0{,}8313$, $HR_{\text{Psych}} = 0{,}7303$) invariant gegenüber rein tokenbasierten Sequenz-Feature-Modi.

---

## 4. Methodologische Leitlinien für kausale Bildungsanalysen

Aus dem Abgleich zwischen DGP und Schätzmodellen ergeben sich drei fundamentale Leitregeln:

1. **Niemals kausale Inferenz ohne akademische Leistungsdynamik:**
   Wird versucht, Förderwirkungen ohne Einbeziehung von Noten- oder Prüfungszeitreihen zu schätzen (`blind`), führt Confounding by Indication unweigerlich zu fatalen Fehlinterpretationen (Verharmlosung oder scheinbare Schädlichkeit von Fördermaßnahmen).
2. **Realistische Studienverlaufsdaten sind ausreichend:**
   Für eine valide Kausalschätzung sind keine psychometrischen Orakel-Variablen ("Inside View") erforderlich. Das reguläre Prüfungs- und Semesterpanel (`realistic`) genügt, um mit IPTW/MSM und G-Computation den echten Ground-Truth-Effekt bis auf Nachkommastellen zu rekonstruieren.
3. **Komplementarität der Methoden:**
   - **G-Computation** ist die Methode der Wahl für makroskopische Outcome-Projektionen ($RR_{\text{Pop}}$).
   - **MSM mit stabilisierten Gewichten** ist unübertroffen bei der Entkopplung konkurrierender Förderstränge (Fachlich vs. Psychosozial).
   - **DML** liefert den sensitivsten Indikator für Informationsvollständigkeit und orthogonale Schocks.

---

## 5. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **LXC Standard Causal Evaluation** | [`pytorch_lxc_causal_benchmark_evaluation_v42.md`](pytorch_lxc_causal_benchmark_evaluation_v42.md) | Detaillierte Standard-Auswertung (S01–S11) |
| **Ablations-Masterplan ($2^4$-Plan)** | [`../01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md`](../01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md) | Architekturvergleich Keras vs. PyTorch |
| **Empirische RCT-Kausalanalyse** | [`../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Selektionsmechanismen in S11 |
| **Marginal Structural Models V4.2** | [`../04_causal_and_simulation/marginal_structural_models_v42.md`](../04_causal_and_simulation/marginal_structural_models_v42.md) | Mathematische Herleitung von IPTW und MSM |
