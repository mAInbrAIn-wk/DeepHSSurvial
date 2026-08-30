import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

sys.path.insert(0, str(Path('src').absolute()))
from autoregressive_next_exam import prepare_next_exam_dataset
from sklearn.preprocessing import StandardScaler

data_dir = Path('src/output_dl_seed99999')
print("Loading data...")
X_hist, X_ctx, y_grades, y_pass, student_ids = prepare_next_exam_dataset(data_dir)

# Group-consistent Split on student IDs
unique_students = np.unique(student_ids)
train_students, temp_students = train_test_split(unique_students, test_size=0.30, random_state=42)
val_students, test_students = train_test_split(temp_students, test_size=0.50, random_state=42)

train_idx = np.where(np.isin(student_ids, train_students))[0]
test_idx = np.where(np.isin(student_ids, test_students))[0]

# Skalierung (like in training)
scaler_seq = StandardScaler()
scaler_ctx = StandardScaler()
v_mask_tr = X_hist[train_idx, :, 0] != -99.0
scaler_seq.fit(X_hist[train_idx][v_mask_tr])
scaler_ctx.fit(X_ctx[train_idx])

X_test_hist = X_hist.copy()[test_idx]
X_test_ctx = X_ctx.copy()[test_idx]

vm_te = X_test_hist[:, :, 0] != -99.0
X_test_hist[vm_te] = scaler_seq.transform(X_test_hist[vm_te])
X_test_ctx = scaler_ctx.transform(X_test_ctx)

y_pass_test = y_pass[test_idx]
y_fail_test = 1 - y_pass_test

print("Loading model...")
model = load_model(data_dir / 'models' / 'autoregressive_next_exam_dual_head.keras')

print("Predicting...")
preds = model.predict({'exam_history': X_test_hist, 'next_exam_context': X_test_ctx}, verbose=0)
pred_pass = preds[1].flatten()
pred_fail = 1.0 - pred_pass

pr_auc_fail = average_precision_score(y_fail_test, pred_fail)
print(f"Next_Exam_Fail_PR_AUC: {pr_auc_fail:.4f}")
print(f"Prevalence of Fail: {np.mean(y_fail_test):.4f}")
