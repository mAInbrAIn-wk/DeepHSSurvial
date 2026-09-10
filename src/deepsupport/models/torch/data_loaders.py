"""
DeepSupport PyTorch Data Layer
==============================
Stellt standardisierte, hochperformante PyTorch Dataset- und DataLoader-
Pipelines für Panel- und Sequenz-Modelle bereit.

Basiert direkt auf dem Data Backbone (feature_builder.py), erzwingt
einen strikten 3-Way Student-Group-Split (70% Train, 15% Val, 15% Test)
ohne Datenleckage und fittet alle Sklearn-Transformer ausschließlich auf Train.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import torch
from torch.utils.data import Dataset, DataLoader

import deepsupport.data_engine.feature_builder as fb


class StudyPanelDataset(Dataset):
    """
    PyTorch Dataset für 2D-Person-Semester- oder Prüfungs-Panels.
    
    Liefert (x, duration, event) bzw. (x, target) für Survival- und Regressionsmodelle.
    """
    def __init__(
        self,
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        competing_events: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        student_ids: Optional[np.ndarray] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.durations = torch.tensor(durations, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)
        self.competing_events = torch.tensor(competing_events, dtype=torch.int64) if competing_events is not None else None
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64) if timesteps is not None else None
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        self.student_ids = student_ids

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.features[idx],
            "duration": self.durations[idx],
            "event": self.events[idx],
        }
        if self.competing_events is not None:
            item["competing_event"] = self.competing_events[idx]
        if self.timesteps is not None:
            item["timestep"] = self.timesteps[idx]
        if self.weights is not None:
            item["weight"] = self.weights[idx]
        return item


class StudySequenceDataset(Dataset):
    """
    PyTorch Dataset für 3D-Sequenzen (N, SeqLen, FeatDim).
    
    Unterstützt Masking für gepaddete Zeitschritte und optionale statische Kontexte.
    """
    def __init__(
        self,
        sequences: np.ndarray,
        targets: Optional[np.ndarray] = None,
        targets_note: Optional[np.ndarray] = None,
        targets_pass: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        student_ids: Optional[np.ndarray] = None,
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        self.targets_note = torch.tensor(targets_note, dtype=torch.float32) if targets_note is not None else None
        self.targets_pass = torch.tensor(targets_pass, dtype=torch.float32) if targets_pass is not None else None
        self.context = torch.tensor(context, dtype=torch.float32) if context is not None else None
        
        if masks is not None:
            self.masks = torch.tensor(masks, dtype=torch.bool)
        else:
            # Automatisches Masking anhand des PADDING_VALUE (-99.0)
            self.masks = (self.sequences[:, :, 0] != fb.PADDING_VALUE)

        self.student_ids = student_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "sequence": self.sequences[idx],
            "mask": self.masks[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.context is not None:
            item["context"] = self.context[idx]
        if self.targets_note is not None:
            item["target_note"] = self.targets_note[idx]
        if self.targets_pass is not None:
            item["target_pass"] = self.targets_pass[idx]
        return item


def prepare_panel_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "standard",
    temporal: str = "prev",
    batch_size: int = 2048,
    num_workers: int = 0,
    seed: int = 42,
    panel_type: str = "semester",
) -> Dict[str, Any]:
    """
    Erstellt vollständig vorkonfigurierte PyTorch-DataLoader für Panel-Daten
    mit striktem Student-Group-Split (70% Train, 15% Val, 15% Test).
    """
    data_path = Path(data_dir)
    
    if panel_type == "semester":
        panel_df, feature_cols, target_col, meta = fb.build_semester_panel_df(
            data_path, mode=mode, temporal=temporal
        )
    elif panel_type == "exam":
        panel_df, feature_cols, target_col, meta = fb.build_exam_panel_df(
            data_path, mode=mode, temporal=temporal
        )
    else:
        raise ValueError(f"Unbekannter panel_type: '{panel_type}'. Erlaubt sind 'semester' oder 'exam'.")

    # Spaltentypen identifizieren
    cat_candidates = ["stg_name", "erstakademiker", "hzb_typ", "migrationshintergrund"]
    cat_cols = [c for c in cat_candidates if c in feature_cols]

    treatment_candidates = ["fach_supp_count", "uebf_supp_count", "psych_supp_count"]
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]

    num_cols = [c for c in feature_cols if c not in cat_cols and c not in treatment_cols]

    # Strikter 3-Way Student Group Split
    unique_studis = np.array(panel_df["studierenden_id"].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    train_panel = panel_df[panel_df["studierenden_id"].isin(train_ids)].copy()
    val_panel = panel_df[panel_df["studierenden_id"].isin(val_ids)].copy()
    test_panel = panel_df[panel_df["studierenden_id"].isin(test_ids)].copy()

    # Preprocessing Pipeline ausschließlich auf Train-Panel fitten
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]),
            cat_cols,
        ))
    if treatment_cols:
        transformers.append((
            "treat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
            ]),
            treatment_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_train = preprocessor.fit_transform(train_panel[feature_cols]).astype(np.float32)
    X_val = preprocessor.transform(val_panel[feature_cols]).astype(np.float32)
    X_test = preprocessor.transform(test_panel[feature_cols]).astype(np.float32)

    # Durations und Events extrahieren
    # Duration = t_stop (Semester), Event = target_col (1 = Dropout/Exmatrikuliert, 0 = Censored/Active)
    t_train = train_panel["t_stop"].values.astype(np.float32)
    e_train = train_panel[target_col].values.astype(np.float32)

    t_val = val_panel["t_stop"].values.astype(np.float32)
    e_val = val_panel[target_col].values.astype(np.float32)

    t_test = test_panel["t_stop"].values.astype(np.float32)
    e_test = test_panel[target_col].values.astype(np.float32)

    # Diskrete Zeitschritte (1 bis 16 Semester, 0-basiert: 0 bis 15 für Hazard-Bins)
    step_train = np.clip(train_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)
    step_val = np.clip(val_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)
    step_test = np.clip(test_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)

    # Competing events (0 = censored, 1 = dropout, 2 = graduation)
    if "competing_event" in train_panel.columns:
        ce_train = train_panel["competing_event"].values.astype(np.int64)
        ce_val = val_panel["competing_event"].values.astype(np.int64)
        ce_test = test_panel["competing_event"].values.astype(np.int64)
    else:
        ce_train = ce_val = ce_test = None

    # Datasets erstellen
    train_dataset = StudyPanelDataset(X_train, t_train, e_train, competing_events=ce_train, timesteps=step_train, student_ids=train_panel["studierenden_id"].values)
    val_dataset = StudyPanelDataset(X_val, t_val, e_val, competing_events=ce_val, timesteps=step_val, student_ids=val_panel["studierenden_id"].values)
    test_dataset = StudyPanelDataset(X_test, t_test, e_test, competing_events=ce_test, timesteps=step_test, student_ids=test_panel["studierenden_id"].values)

    # Reproduzierbare Worker-Initialisierung
    def _seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "input_dim": X_train.shape[1],
        "X_train": X_train,
        "t_train": t_train,
        "e_train": e_train,
        "ce_train": ce_train,
        "X_val": X_val,
        "t_val": t_val,
        "e_val": e_val,
        "ce_val": ce_val,
        "X_test": X_test,
        "t_test": t_test,
        "e_test": e_test,
        "ce_test": ce_test,
        "train_panel": train_panel,
        "val_panel": val_panel,
        "test_panel": test_panel,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "treatment_cols": treatment_cols,
        "target_col": target_col,
    }


def prepare_sequence_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "standard",
    temporal: str = "prev",
    batch_size: int = 512,
    num_workers: int = 0,
    seed: int = 42,
    seq_type: str = "exam",
    max_len: int = 40,
) -> Dict[str, Any]:
    """
    Erstellt PyTorch DataLoader für 3D-Sequenzmodelle (Prüfungsverläufe oder Semesterfolgen).
    Group-Split erfolgt strikt auf Studierenden-Ebene.
    """
    data_path = Path(data_dir)
    df_abschluesse, _ = fb._load_raw_data(data_path)
    note_dict = df_abschluesse.set_index("studierenden_id")["abschlussnote"].to_dict()
    status_dict = df_abschluesse.set_index("studierenden_id")["status"].to_dict()

    if seq_type == "exam":
        studis, X_seq, _, _, feat_names, _ = fb.build_exam_sequence_tensor(
            data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="grade"
        )
    elif seq_type == "semester":
        studis, X_seq, _, _, feat_names, _ = fb.build_semester_sequence_tensor(
            data_path, max_semesters=16, mode=mode, temporal=temporal, target_type="gpa"
        )
    else:
        raise ValueError(f"Unbekannter seq_type: '{seq_type}'. Erlaubt sind 'exam' oder 'semester'.")

    # Target: Abschlussnote (kontinuierlich) und Dropout/Abschluss (binär)
    y_note_all = np.array([note_dict.get(s, np.nan) for s in studis], dtype=np.float32)
    # 1 für Dropout, 0 für Abschluss
    dropout_statuses = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}
    y_drop_all = np.array([1.0 if str(status_dict.get(s, "")).strip().lower() in dropout_statuses else 0.0 for s in studis], dtype=np.float32)

    # Strikter Student-Level Split (70/15/15) auf allen Studierenden
    unique_studis = np.unique(studis)
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    train_mask = np.isin(studis, train_ids)
    val_mask = np.isin(studis, val_ids)
    test_mask = np.isin(studis, test_ids)

    train_seq = X_seq[train_mask]
    val_seq = X_seq[val_mask]
    test_seq = X_seq[test_mask]

    # Preprocessing: StandardScaler nur auf unpadded Werten des Train-Splits
    v_mask_tr = (train_seq[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(train_seq[v_mask_tr])

    def _transform_seq(seq_arr: np.ndarray) -> np.ndarray:
        out = seq_arr.copy()
        vm = (seq_arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(seq_arr[vm])
        return out

    train_seq_norm = _transform_seq(train_seq).astype(np.float32)
    val_seq_norm = _transform_seq(val_seq).astype(np.float32)
    test_seq_norm = _transform_seq(test_seq).astype(np.float32)

    train_note = y_note_all[train_mask]
    val_note = y_note_all[val_mask]
    test_note = y_note_all[test_mask]

    train_drop = y_drop_all[train_mask]
    val_drop = y_drop_all[val_mask]
    test_drop = y_drop_all[test_mask]

    train_dataset = StudySequenceDataset(train_seq_norm, targets_note=train_note, targets_pass=train_drop, student_ids=studis[train_mask])
    val_dataset = StudySequenceDataset(val_seq_norm, targets_note=val_note, targets_pass=val_drop, student_ids=studis[val_mask])
    test_dataset = StudySequenceDataset(test_seq_norm, targets_note=test_note, targets_pass=test_drop, student_ids=studis[test_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "seq_len": X_seq.shape[1],
        "feature_dim": X_seq.shape[2],
        "scaler": scaler,
        "feature_names": feat_names,
        "y_note_test": test_note,
        "y_drop_test": test_drop,
        "y_note_val": val_note,
        "y_drop_val": val_drop,
    }


def prepare_exam_regressor_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 256,
    seed: int = 42,
    max_len: int = 40,
) -> Dict[str, Any]:
    """
    Bereitet Daten für den PyTorchExamTransformerRegressor vor:
    - Ausschließlich Absolventen (Studierende mit gültiger Abschlussnote)
    - Default-Modus: gradeblind (strikt ohne Notenmerkmale)
    - Group-konsistenter Student-Split (70/15/15)
    - StandardScaler gefittet nur auf Train-Split (unpadded Schritte)
    """
    data_path = Path(data_dir)
    studis, X_seq, _, _, feat_names, _ = fb.build_exam_sequence_tensor(
        data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="grade"
    )
    df_abschluesse, _ = fb._load_raw_data(data_path)
    note_dict = df_abschluesse.set_index("studierenden_id")["abschlussnote"].to_dict()
    y_all = np.array([note_dict.get(s, np.nan) for s in studis], dtype=np.float32)

    # Filtere strikt auf Absolventen
    valid_grad = ~np.isnan(y_all)
    studis_clean = studis[valid_grad]
    X_clean = X_seq[valid_grad]
    y_clean = y_all[valid_grad]

    unique_studis = np.unique(studis_clean)
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    train_mask = np.isin(studis_clean, train_ids)
    val_mask = np.isin(studis_clean, val_ids)
    test_mask = np.isin(studis_clean, test_ids)

    X_train = X_clean[train_mask].copy()
    X_val = X_clean[val_mask].copy()
    X_test = X_clean[test_mask].copy()

    y_train = y_clean[train_mask]
    y_val = y_clean[val_mask]
    y_test = y_clean[test_mask]

    # Preprocessing: StandardScaler auf unpadded Train-Tokens
    vm_tr = (X_train[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(X_train[vm_tr])

    def _transform(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(arr[vm])
        return out.astype(np.float32)

    X_train_norm = _transform(X_train)
    X_val_norm = _transform(X_val)
    X_test_norm = _transform(X_test)

    train_ds = StudySequenceDataset(X_train_norm, targets=y_train, student_ids=studis_clean[train_mask])
    val_ds = StudySequenceDataset(X_val_norm, targets=y_val, student_ids=studis_clean[val_mask])
    test_ds = StudySequenceDataset(X_test_norm, targets=y_test, student_ids=studis_clean[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test": X_test_norm,
        "y_test": y_test,
        "test_mask": (X_test_norm[:, :, 0] != fb.PADDING_VALUE),
        "feat_names": feat_names,
        "feature_dim": X_clean.shape[2],
        "scaler": scaler,
        "n_graduates_test": len(y_test),
    }


def prepare_causal_survival_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 256,
    seed: int = 42,
    max_len: int = 40,
) -> Dict[str, Any]:
    """
    Bereitet Daten für PyTorchCausalExamTransformerSurvival vor:
    - Gesamte Kohorte (Absolventen + Abbrecher)
    - Kausales Prüfungsschritt-Target y_seq (N, max_len, 1)
    - 70/15/15 Split stratifiziert nach Endstatus (Dropout vs. Abschluss)
    - StandardScaler auf unpadded Train-Tokens
    """
    data_path = Path(data_dir)
    studis, X_seq, y_seq, studi_events, feat_names, _ = fb.build_exam_sequence_tensor(
        data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="dropout"
    )

    # 3-Way Stratified Split auf Student-Ebene
    unique_studis = np.unique(studis)
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed, stratify=studi_events)
    temp_mask = np.isin(studis, temp_ids)
    temp_events = studi_events[temp_mask]
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed, stratify=temp_events)

    train_mask = np.isin(studis, train_ids)
    val_mask = np.isin(studis, val_ids)
    test_mask = np.isin(studis, test_ids)

    X_train = X_seq[train_mask].copy()
    X_val = X_seq[val_mask].copy()
    X_test = X_seq[test_mask].copy()

    y_train = y_seq[train_mask].squeeze(-1)  # (N, S)
    y_val = y_seq[val_mask].squeeze(-1)
    y_test = y_seq[test_mask].squeeze(-1)

    # Preprocessing: StandardScaler auf unpadded Train-Tokens
    vm_tr = (X_train[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(X_train[vm_tr])

    def _transform(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(arr[vm])
        return out.astype(np.float32)

    X_train_norm = _transform(X_train)
    X_val_norm = _transform(X_val)
    X_test_norm = _transform(X_test)

    train_ds = StudySequenceDataset(X_train_norm, targets=y_train, student_ids=studis[train_mask])
    val_ds = StudySequenceDataset(X_val_norm, targets=y_val, student_ids=studis[val_mask])
    test_ds = StudySequenceDataset(X_test_norm, targets=y_test, student_ids=studis[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test": X_test_norm,
        "y_test": y_test,
        "test_mask": (X_test_norm[:, :, 0] != fb.PADDING_VALUE),
        "test_student_events": studi_events[test_mask],
        "feat_names": feat_names,
        "feature_dim": X_seq.shape[2],
        "scaler": scaler,
        "n_students_test": len(test_ids),
    }
