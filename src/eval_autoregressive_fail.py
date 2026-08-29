import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

sys.path.insert(0, str(Path('src').absolute()))
import feature_builder as fb

data_dir = Path('src/output_dl_seed99999')
print("Loading data...")
X_seq, y_grades, y_pass = fb.build_autoregressive_dataset(data_dir, max_exams=50)

idx = np.arange(len(y_grades))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

X_test = X_seq[test_idx]
y_pass_test = y_pass[test_idx]
y_fail_test = 1 - y_pass_test

print("Loading model...")
model = load_model(data_dir / 'models' / 'autoregressive_next_exam_dual_head.keras')

print("Predicting...")
preds = model.predict(X_test, verbose=0)
pred_pass = preds[1].flatten()
pred_fail = 1.0 - pred_pass

pr_auc_fail = average_precision_score(y_fail_test, pred_fail)
print(f"Next_Exam_Fail_PR_AUC: {pr_auc_fail:.4f}")
print(f"Prevalence of Fail: {np.mean(y_fail_test):.4f}")
