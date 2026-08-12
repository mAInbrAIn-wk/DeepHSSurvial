import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
from config import CONFIG
from models import Student

def as_dataframe(studierende: List[Student], stammdaten: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    studierende_rows = []
    einschreibungen_rows = []
    pruefungen_rows = []
    support_teilnahmen_rows = []
    abschluesse_rows = []

    # Map modul_id -> modul name for Bachelorarbeit lookup
    ba_modul_ids = set()
    if stammdaten is not None and "module_df" in stammdaten:
        module_df = stammdaten["module_df"]
        ba_modul_ids = set(module_df[module_df["name"].str.lower().str.contains("bachelorarbeit")]["modul_id"])

    for s in studierende:
        studierende_rows.append({
            "studierenden_id": s.studierenden_id,
            "studiengang_id": s.studiengang_id,
            "kohorten_semester_id": s.kohorten_semester_id,
            "geschlecht": s.geschlecht,
            "alter_immatrikulation": s.alter_immatrikulation,
            "hzb_note": s.hzb_note,
            "hzb_typ": s.hzb_typ,
            "migrationshintergrund": s.migrationshintergrund,
            "erstakademiker": s.erstakademiker,
            "erwerbstaetigkeit_std": s.erwerbstaetigkeit_std,
            "motivation_initial": s.motivation_initial,
            "soziale_integration_initial": s.soziale_integration_initial,
            "motivation_final": round(s.motivation, 3),
            "soziale_integration_final": round(s.soziale_integration, 3),
            "hidden_erwartete_note_initial": s.erwartete_note_initial,
            "hidden_erwartete_note_final": round(s.erwartete_note, 3),
            "hidden_zeit_puffer": round(getattr(s, 'hidden_zeit_puffer', 60.0), 1),
        })
        
        for e in s.einschreibungen:
            e["studierenden_id"] = s.studierenden_id
            einschreibungen_rows.append(e)
            
        for st in s.support_teilnahmen:
            st["studierenden_id"] = s.studierenden_id
            support_teilnahmen_rows.append(st)
            
        for p in s.pruefungen:
            pruefungen_rows.append({
                "studierenden_id": s.studierenden_id,
                "semester_id": p.semester_id,
                "modul_id": p.modul_id,
                "versuch": p.versuch,
                "note": p.note,
                "bestanden": p.bestanden,
                "note_counterfactual": p.note_counterfactual,
                "support_genutzt": p.support_genutzt,
                "hidden_motivation": round(p.hidden_motivation, 3) if p.hidden_motivation is not None else None,
                "hidden_soziale_integration": round(p.hidden_soziale_integration, 3) if p.hidden_soziale_integration is not None else None,
                "hidden_erwartete_note": round(p.hidden_erwartete_note, 3) if p.hidden_erwartete_note is not None else None,
                "hidden_overload": round(p.hidden_overload, 1) if p.hidden_overload is not None else 0.0,
                "hidden_zeit_puffer": round(p.hidden_zeit_puffer, 1) if p.hidden_zeit_puffer is not None else 60.0,
                "hidden_penalty_capped": bool(p.hidden_penalty_capped) if p.hidden_penalty_capped is not None else False,
                "hidden_support_capped": bool(p.hidden_support_capped) if p.hidden_support_capped is not None else False,
            })
            
        status = "abgeschlossen" if s.abschluss_erreicht else ("exmatrikuliert" if s.exmatrikuliert else ("abgebrochen" if s.abgebrochen else "zeitueberschreitung"))
        letztes_sem = s.einschreibungen[-1] if s.einschreibungen else None
        
        # Abschlussnote & Bachelorarbeitsnote NUR bei abgeschlossenem Studium berechnen
        abschlussnote = None
        bachelorarbeitsnote = None
        
        if status == "abgeschlossen":
            bestandene = [p for p in s.pruefungen if p.bestanden]
            letzte_versuche = {}
            for p in bestandene:
                letzte_versuche[p.modul_id] = p.note
                if p.modul_id in ba_modul_ids:
                    bachelorarbeitsnote = p.note
                    
            if letzte_versuche:
                abschlussnote = round(sum(letzte_versuche.values()) / len(letzte_versuche), 2)
        
        abschluesse_rows.append({
            "studierenden_id": s.studierenden_id,
            "status": status,
            "abschluss_semester_id": letztes_sem["semester_id"] if letztes_sem else None,
            "studiendauer_semester": letztes_sem["fachsemester"] if letztes_sem else 0,
            "abschlussnote": abschlussnote,
            "bachelorarbeitsnote": bachelorarbeitsnote,
            "anomalie_typ": s.anomalie_typ
        })

    return {
        "studierende_df": pd.DataFrame(studierende_rows),
        "einschreibungen_df": pd.DataFrame(einschreibungen_rows),
        "pruefungen_df": pd.DataFrame(pruefungen_rows),
        "support_teilnahmen_df": pd.DataFrame(support_teilnahmen_rows, columns=["studierenden_id", "semester_id", "angebot_id"]),
        "abschluesse_df": pd.DataFrame(abschluesse_rows),
    }

def exportiere_csv(daten: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df in daten.items():
        if df is not None:
            pfad = output_dir / f"{key.replace('_df', '')}.csv"
            df.to_csv(pfad, index=False, sep=",", decimal=".")
            print(f"  [OK] {pfad.name:<25} {len(df):>8} Zeilen")
