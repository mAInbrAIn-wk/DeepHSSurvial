"""
DEPRECATED / CONSOLIDATED WRAPPER: extended_cox_delta.py
=========================================================
Dieses Skript wurde in `src/extended_cox_survival.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extended_cox_survival import train_extended_cox_model

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_extended_cox_model(data_dir, temporal='prev', mode='standard')
