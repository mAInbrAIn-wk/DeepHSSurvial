# V3 → V4 Regression: Verlorene RNG-Synchronisierung

## Zusammenfassung

> [!CAUTION]
> Beim V4-Refactoring sind **drei kritische RNG-Schutzmechanismen** aus V3 verloren gegangen. Die gesamte V4-Simulation hat desynchronisierte Zufallsströme zwischen den Universen.

---

## Vollständiger Diff: V3 vs V4 RNG-Architektur

### 1. Per-Student-Seeds (V3 ✅ → V4 ❌)

**V3** ([`simulation_v3.py` Z. 108–112](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L108-L112)):
```python
for idx, studi in enumerate(studierende):
    base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ population_seed) & 0xFFFFFFFF
    rng_init    = np.random.default_rng(base_seed)
    rng_support = np.random.default_rng((base_seed + 1) & 0xFFFFFFFF)
    rng_social  = np.random.default_rng((base_seed + 2) & 0xFFFFFFFF)
    rng_dropout = np.random.default_rng((base_seed + 3) & 0xFFFFFFFF)
```

**V4** ([`simulation_v4.py` Z. 222–226](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v4.py#L222-L226)):
```python
for idx, studi in enumerate(studierende):
    studi.stat_modules_dropped = 0
    # ... keine Per-Student-Seeds, nur ein globaler rng
```

**Was verloren ging:** Jeder Student hatte in V3 seine eigenen, voneinander unabhängigen RNG-Generatoren. In V4 teilen sich **alle 25.000 Studierenden einen einzigen RNG-Stream** — jede bedingte Ziehung bei Student $k$ verschiebt den Stream für Student $k+1$.

### 2. Deterministisches Prüfungsrauschen (V3 ✅ → V4 ❌)

**V3** ([`simulation_v3.py` Z. 11–13 + Z. 247](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L11-L13)):
```python
def get_exam_noise(base_seed: int, modul_id: str, versuch: int) -> float:
    exam_seed = (base_seed ^ zlib.crc32(f"{modul_id}_{versuch}".encode())) & 0xFFFFFFFF
    return float(np.random.default_rng(exam_seed).normal(0, CONFIG["gewicht_rauschen"]))

# Aufruf (Z. 247):
e_noise = get_exam_noise(base_seed, m_id, m_state.versuche)
```

**V4** ([`simulation_v4.py` Z. 152](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v4.py#L152)):
```python
rng.normal(0, cfg["gewicht_rauschen"])  # Globaler sequentieller Stream
```

**Was verloren ging:** In V3 war das Prüfungsrauschen eine **reine Funktion von (Student-ID, Modul-ID, Versuch)** — absolut deterministisch und positionsunabhängig. In V4 ist es ein sequentieller Draw aus dem globalen Stream — jede vorherige Support-Entscheidung verschiebt das Rauschen aller folgenden Prüfungen.

### 3. Pad-Draws für blockierte Angebote (V3 ✅ → V4 ❌)

**V3** ([`simulation_v3.py` Z. 193–205](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L193-L205)):
```python
nutzt_support = rng_support.random() < p          # IMMER ziehen
blocked = (typ == "fachlich" and block_fach) or ...
if nutzt_support and not blocked:
    ...
elif nutzt_support and blocked:
    if verfuegbare_zeit - ... < 0:
        _ = rng_support.random()                   # Pad-Draw für Zeitcheck
```

**V4** ([`simulation_v4.py` Z. 297–335](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v4.py#L297-L335)):
```python
for angebot in support_list:                        # support_list ist VORHER gefiltert!
    ...
    if rng.random() < p:                            # Nur wenn Angebot nicht blockiert
```

**Was verloren ging:** V3 iterierte über **alle** Angebote und zog **immer** eine Zufallszahl (auch für blockierte). V4 filtert die Angebote **vorher** aus der Liste, sodass in Uni B überhaupt keine Draws stattfinden.

---

## Was noch alles „passiert" ist

### 4. Carry-over-Effekt des fachlichen Supports (V3 ✅ → V4 ❌)

**V3** (Z. 238–241):
```python
carryover_ids = bisherige_fach_supports - set(teilgenommene_angebote)
carryover_boost_sum = sum(...) * (2.0 / 3.0)    # 2/3 Wirkung aus Vorsemestern
```

**V4:** Kein Carry-over. Fachlicher Support wirkt nur im Semester der Teilnahme.

### 5. Overload-Penalty Cap (V3 ✅ → V4 geändert)

**V3** (Z. 230): `overload_penalty = float(min(0.15, (overload / 100.0) * 0.1))`  
**V4** (Z. 369): `overload_penalty = (overload / 100.0) * overload_penalty_factor` (kein Cap, unbegrenzt in Noten)

### 6. Soziale Integration Drift (V3 ✅ → V4 ❌)

**V3** (Z. 300): `studi.soziale_integration += rng_social.normal(0, 0.05)` — separater Stream  
**V4:** Kein soziales Driften zwischen Semestern (Wert ändert sich nur durch Support-Teilnahme)

---

## Reparaturplan

### Phase 1: RNG-Streams reparieren (Priorität: KRITISCH)

In `simulation_v4.py` die V3-Architektur wiederherstellen:

```python
import zlib

def get_exam_noise(base_seed: int, modul_id: str, versuch: int, cfg) -> float:
    exam_seed = (base_seed ^ zlib.crc32(f"{modul_id}_{versuch}".encode())) & 0xFFFFFFFF
    return float(np.random.default_rng(exam_seed).normal(0, cfg["gewicht_rauschen"]))

# In simuliere_verlaeufe, pro Student:
for idx, studi in enumerate(studierende):
    base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ population_seed) & 0xFFFFFFFF
    rng_support = np.random.default_rng((base_seed + 1) & 0xFFFFFFFF)
    rng_social  = np.random.default_rng((base_seed + 2) & 0xFFFFFFFF)
    rng_dropout = np.random.default_rng((base_seed + 3) & 0xFFFFFFFF)
```

### Phase 2: Support-Schleife über ALLE Angebote iterieren

Statt `support_list` (vorfiltriert) über die vollständige Liste iterieren und blockierte Angebote **nach** dem Draw ignorieren — wie in V3.

### Phase 3: Carry-over und soziale Drift entscheiden

Soll der Carry-over-Effekt und die soziale Drift zurück? Das sind inhaltliche Entscheidungen.

### Phase 4: Re-Run aller Universen und Grid-Szenarien

> [!IMPORTANT]
> **Alle bisherigen V4-Ergebnisse (Dropout-Raten, Relative Risiken, Schutzeffekte, Migrationsanalysen) sind durch die RNG-Desynchronisierung kontaminiert.** Die Makro-Trends (Support hilft, Erstakademiker profitieren mehr) sind qualitativ korrekt, aber die exakten Zahlen sind unzuverlässig. Ein Re-Run nach der Reparatur ist zwingend nötig.
