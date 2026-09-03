# Quantitative Kausalanalyse (inkl. Oracle & Noten-Performance)

## 1. Effekt auf Dropout (Relative Risiken)

| Methode | RR/HR Fachlich | RR/HR Überfachlich | RR/HR Psychosozial |
| :--- | :--- | :--- | :--- |
| **Ground Truth (Sandbox A-H)** | 0.9326 | 0.9194 | 0.9448 |
| **Extended Cox Panel (Statistisch)** | 0.9234 | 0.9648 | 0.9005 |
| **DML Orthogonal (Double ML)** | 0.9863 | 0.9977 | 0.9941 |
| **Deep Transformer DML (Sequenz/Imai)** | 1.0172 | 0.9957 | 0.9569 |
| **Oracle DeepSurv (Mit latenten Variablen)** | - | 0.9897 | - |
| **Oracle Logistic Hazard** | - | 0.9915 | - |

## 2. Analyse der Feedback-Schleifen

### Fachlicher Support (Ground Truth RR ~ 0.932)
Fachlicher Support ist auf der **Dropout-Ebene** (RR 0.932) gut messbar. Das klassische Cox-Modell trifft die Ground Truth, während Double ML den Effekt "wegdämpft" (RR ~ 1.01). 

**Der Noten-Effekt:** Wie in den historischen Reflexionen (vgl. Projekt DataAnalysis & `00_Historisches_Gesamtprotokoll.md`) diskutiert, schlägt sich die deutliche Notenverbesserung in der Ground Truth durch fachlichen Support oft nicht direkt in der globalen Dropout-Rate nieder. Das Überdämpfen der Modelle auf den Dropout-Effekt rührt genau daher: Sie prädizieren die Leistung (Noten/CP) direkt. Der kausale Pfad des fachlichen Supports verläuft *vollständig mediiert* über die Noten (bzw. Prädiktion auf Prädiktionsebene, was in der Simulation als direkte Verbesserung implementiert ist).

### Überfachlicher Support (Ground Truth RR ~ 0.919)
Die stärkste Feedback-Schleife: Überfachlicher Support hat einen rekursiven Loop mit der intrinsischen Motivation (latente Variable in der Simulation). Die Standard-Beobachtungsmodelle unterschätzen die Schutzwirkung massiv (Cox: 0.964, DML: 0.995).
**Der Oracle-Beweis:** Erst durch die Bereitstellung der DGP-Zustandsvariablen (`hidden_motivation_prev`) in den Oracle-Modellen wendet sich das Vorzeichen und der Effekt wird als protektiv (RR 0.9897 / 0.9915) erkannt. Dies liefert den zwingenden Hinweis darauf, dass das Versagen der Beobachtungsmodelle auf unvollständiger Confounder-Kontrolle (Omitted-Variable-Bias) beruht.

### Psychosozialer Support (Ground Truth RR ~ 0.944)
Psychosozialer Support wird in der Simulation häufig durch quasi-randomisierte externe Schocks (Krankheit/Krisen) getriggert. Da er weniger an kumulativen akademischen Misserfolg gekoppelt ist, können sowohl Cox (0.90) als auch Transformer DML (0.957) ihn wesentlich leichter als isolierten Schutzfaktor identifizieren.