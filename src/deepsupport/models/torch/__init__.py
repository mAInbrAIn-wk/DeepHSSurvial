"""
DeepSupport PyTorch & PyCox Modeling Suite
==========================================
Enthält native PyTorch- und PyCox-Implementierungen für Survival-Analyse,
Sequenzmodellierung und autoregressive Transformer.
"""

from deepsupport.models.torch.data_loaders import (
    StudyPanelDataset,
    StudySequenceDataset,
    prepare_panel_dataloaders,
    prepare_sequence_dataloaders,
    prepare_exam_regressor_dataloaders,
    prepare_causal_survival_dataloaders,
)
from deepsupport.models.torch.survival import (
    MLPBackbone,
    PyTorchLogisticHazard,
    PyTorchDeepHit,
    PyTorchCoxPH,
    PyTorchCoxTime,
    PyTorchDeepHitCompetingRisks,
)
from deepsupport.models.torch.transformer import (
    SinCosPositionalEncoding,
    AttentionPooling,
    TransformerEncoderBlock,
    PyTorchExamTransformerRegressor,
    PyTorchCausalExamTransformerSurvival,
    PyTorchNextExamTransformer,
)
from deepsupport.models.torch.trainer import (
    ModelTrainer,
    EarlyStopping,
)

__all__ = [
    "StudyPanelDataset",
    "StudySequenceDataset",
    "prepare_panel_dataloaders",
    "prepare_sequence_dataloaders",
    "prepare_exam_regressor_dataloaders",
    "prepare_causal_survival_dataloaders",
    "MLPBackbone",
    "PyTorchLogisticHazard",
    "PyTorchDeepHit",
    "PyTorchCoxPH",
    "PyTorchCoxTime",
    "PyTorchDeepHitCompetingRisks",
    "SinCosPositionalEncoding",
    "AttentionPooling",
    "TransformerEncoderBlock",
    "PyTorchExamTransformerRegressor",
    "PyTorchCausalExamTransformerSurvival",
    "PyTorchNextExamTransformer",
    "ModelTrainer",
    "EarlyStopping",
]
