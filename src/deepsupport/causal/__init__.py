"""
DeepSupport PyTorch Causal Suite
================================
Enthält alle kausalen Schätzverfahren in nativem PyTorch:
- PyTorchDMLSurvival: Double Machine Learning mit 5-Fold Student Cross-Fitting und Robinson-Schätzer
- PyTorchMSM: Marginal Structural Models mit zeitvariierenden IPTW-Gewichten und Cluster-Robuster Kovarianz
- PyTorchGComputation: Vektorisierte G-Formel Kontrafaktik-Simulation unter do(A=0) vs. do(A=1)
"""

from deepsupport.causal.torch_dml import PyTorchDMLSurvival
from deepsupport.causal.torch_msm import PyTorchMSM
from deepsupport.causal.torch_gcomputation import PyTorchGComputation

__all__ = [
    "PyTorchDMLSurvival",
    "PyTorchMSM",
    "PyTorchGComputation",
]
