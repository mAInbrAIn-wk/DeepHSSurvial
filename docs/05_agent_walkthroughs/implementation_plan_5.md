# Implementation Plan: Simulation V3.2 (Support-Boost Feintuning)

## Zielsetzung
Der fachliche Support-Boost soll verdoppelt werden, um den positiven Effekt auf die Noten deutlicher herauszustellen und ihn stärker gegen die Overload-Penalty zu gewichten. Zusätzlich soll diagnostisches Logging eingeführt werden, um zu überprüfen, wie oft der Support-Boost durch den `support_deckel` (Cap) nach oben begrenzt wird.

## Klärung der offenen Fragen aus dem Review

> [!NOTE]
> **1. Warum schützte Modulabwurf vor Dropout (CP-Rückstand vs. Aktuell Durchgefallen)?**
> Ein Modulabwurf erhöht zwar den `cp_rueckstand`. Ein fehlendes Modul (5 CP) erhöht die Dropout-Wahrscheinlichkeit um `5/30 * 0.15 * 0.5 = +1.25%`. 
> Behält man das Modul aber bei und *fällt durch die Prüfung*, erhöht das `durchgefallen_aktuell`, was die Dropout-Wahrscheinlichkeit sofort um `1 * 0.04 * 0.5 = +2.00%` erhöht! Der Modulabwurf ist also die mathematisch "sicherere" Variante, solange man nicht in die endgültige Zeitüberschreitung läuft.

> [!NOTE]
> **2. Warum ist die Support-Nutzung in V3.1 so stark gestiegen?**
> In V2 dauerte ein abgebrochenes Studium im Durchschnitt **4.50 Semester**, bevor der Dropout passierte. In V3.1 bleiben die Abbrecher im Schnitt **5.31 Semester** im System, da der schützende (aber falsche) Modulabwurf-Mechanismus fehlt und sie sich "länger durchschleppen". Ein fast komplettes zusätzliches Semester an der Uni bedeutet automatisch auch ein Semester mehr Zeit, in dem sie Supports in Anspruch nehmen! Daher der Anstieg von 144k auf 148k.

---

## Proposed Changes

### 1. Konfiguration (`config.py`)

#### [MODIFY] [config.py](file:///c:/GitHub_public/Abschlussprojekt/src/config.py)
Wir verdoppeln das Gewicht des Boosts und heben gleichzeitig den Deckel an, damit der erhöhte Boost nicht sofort künstlich beschnitten wird.

```diff
-    'gewicht_support_boost': 0.04,
+    'gewicht_support_boost': 0.08,  # V3.2: Support Boost verdoppelt
     'integration_startwert': 0.65,
     'leistung_startwert': 0.55,
     'seed': 42,
     'start_jahr': 2015,
-    'support_deckel': 1.0,
+    'support_deckel': 2.0,  # V3.2: Deckel erhöht, passend zum verdoppelten Boost
```

### 2. Datenmodelle (`models.py`)

#### [MODIFY] [models.py](file:///c:/GitHub_public/Abschlussprojekt/src/models.py)
Erweiterung der `PruefungsErgebnis`-Datenklasse um das neue Cap-Tracking.

```diff
 @dataclass
 class PruefungsErgebnis:
     semester_id: int
     modul_id: str
     versuch: int
     note: float
     bestanden: bool
     note_counterfactual: float
     support_genutzt: bool
     hidden_motivation: float
     hidden_soziale_integration: float
     hidden_erwartete_note: float
     hidden_overload: float = 0.0
     hidden_zeit_puffer: float = 60.0
     hidden_penalty_capped: bool = False
+    hidden_support_capped: bool = False  # V3.2: Wurde der Boost durch den Deckel begrenzt?
```

### 3. Simulation (`simulation_v3.py`)

#### [MODIFY] [simulation_v3.py](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v3.py)
Wir berechnen den ungedeckeltem Boost und prüfen, ob er den Deckel übersteigt. Anschließend exportieren wir das neue Feld im CSV.

```diff
-                boost = float(np.clip(boost_sum * CONFIG["gewicht_support_boost"] * CONFIG.get("support_effect_multiplier", 1.0), 0.0, CONFIG["support_deckel"])) if boost_sum > 0.0 else 0.0
+                # V3.2: Tracking des Support-Deckels
+                raw_boost = boost_sum * CONFIG["gewicht_support_boost"] * CONFIG.get("support_effect_multiplier", 1.0) if boost_sum > 0.0 else 0.0
+                boost = float(np.clip(raw_boost, 0.0, CONFIG["support_deckel"]))
+                support_capped = (raw_boost > CONFIG["support_deckel"])
                 
                 note, bestanden, note_cf = simuliere_pruefung(
                     schwierigkeit=modul_data[m_id]["schwierigkeit"],
                     erwartete_note=studi.erwartete_note,
                     motivation=studi.motivation,
                     soz_int=studi.soziale_integration,
                     fachlicher_boost=boost,
                     versuch=m_state.versuche,
                     overload_penalty=overload_penalty,
                     rng=rng
                 )
                 
                 studi.pruefungen.append(PruefungsErgebnis(
                     semester_id=akt_sem_id, modul_id=m_id, versuch=m_state.versuche, 
                     note=note, bestanden=bestanden, note_counterfactual=note_cf, support_genutzt=(boost > 0),
                     hidden_motivation=studi.motivation,
                     hidden_soziale_integration=studi.soziale_integration,
                     hidden_erwartete_note=studi.erwartete_note,
                     hidden_overload=overload,
                     hidden_zeit_puffer=puffer,
-                    hidden_penalty_capped=(overload_penalty >= 0.15)
+                    hidden_penalty_capped=(overload_penalty >= 0.15),
+                    hidden_support_capped=support_capped
                 ))
```

Zudem muss der CSV-Export-Block für `pruefungen` (Zeile 300) um die Spalte `hidden_support_capped` erweitert werden.

## Verification Plan
1. **Durchführen der Anpassungen** in den jeweiligen Dateien.
2. **Starten der Simulation** über `simulation_v3.py`.
3. **Auswertung des Deckels:** Ein kurzes Skript prüft, wie viel Prozent der Prüfungen mit Support-Nutzung das Flag `hidden_support_capped == True` aufweisen.
4. **Migrationsanalyse G1/G2:** Wir überprüfen, ob der erhöhte Boost die Anzahl der "echten Geretteten" in V3.2 im Vergleich zu V3.1 signifikant ansteigen lässt.

> [!WARNING]
> Mit diesem Patch sind wir bereit für V3.2! Wenn der Plan so in Ordnung ist, bestätigen Sie kurz, und ich setze die Änderungen um.
