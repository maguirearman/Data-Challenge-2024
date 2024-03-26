import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Load the training data
train_demos_df = pd.read_csv("train/train_demos.csv")
train_signs_df = pd.read_csv("train/train_signs.csv")
train_radiology_df = pd.read_csv("train/train_radiology.csv")
train_labels_df = pd.read_csv("train/train_labels.csv")

#Load the testing data
test_demos_df = pd.read_csv("test/test_demos.csv")
test_signs_df = pd.read_csv("test/test_signs.csv")
test_radiology_df = pd.read_csv("test/test_radiology.csv")

# Define thresholds
healthy_hr_min, healthy_hr_max = 60, 100
healthy_resp_min, healthy_resp_max = 12, 25
healthy_nbp_sys_min, healthy_nbp_sys_max = 90, 140
spo2_threshold = 90
# Process training data to calculate flags
flags_df = train_signs_df.groupby('patient_id').agg({
    'heartrate': lambda hr: ((hr < healthy_hr_min) | (hr > healthy_hr_max)).any().astype(int),
    'resp': lambda r: ((r < healthy_resp_min) | (r > healthy_resp_max)).any().astype(int),
    'nbpsys': lambda nbp: ((nbp < healthy_nbp_sys_min) | (nbp > healthy_nbp_sys_max)).any().astype(int),
    'spo2': lambda s: (s < spo2_threshold).any().astype(int)
}).reset_index()

# Merge flags with labels
features_df = pd.merge(flags_df, train_labels_df, on='patient_id', how='left')

# Prepare features and labels for modeling
X = features_df[['heartrate', 'resp', 'nbpsys', 'spo2']].to_numpy()
y = features_df['label'].to_numpy().astype(int)

# Split the data for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred_proba = model.predict_proba(X_val)[:, 1]
val_auroc_score = roc_auc_score(y_val, y_val_pred_proba)
print("Validation AUROC Score:", val_auroc_score)