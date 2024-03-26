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

# Define healthy range for heart rate and respiratory rate
healthy_hr_min, healthy_hr_max = 60, 100
healthy_resp_min, healthy_resp_max = 12, 25
# Define healthy range for NBP Sys
healthy_nbp_sys_min, healthy_nbp_sys_max = 90, 140
# Define a threshold for SpO2. Values below this might be considered low
spo2_threshold = 95

# Initialize a dictionary to store patient flags
patient_flags = {}

# Process each patient's data
for patient_id in train_demos_df['patient_id'].unique():
    patient_data = train_signs_df[train_signs_df['patient_id'] == patient_id]
    hr_flag = resp_flag = nbp_sys_flag = spo2_flag = 0
    
    # Existing checks for heart rate and respiratory rate
    if not patient_data[patient_data['heartrate'].notna() & ((patient_data['heartrate'] < healthy_hr_min) | (patient_data['heartrate'] > healthy_hr_max))].empty:
        hr_flag = 1
    if not patient_data[patient_data['resp'].notna() & ((patient_data['resp'] < healthy_resp_min) | (patient_data['resp'] > healthy_resp_max))].empty:
        resp_flag = 1
    # New check for NBP Sys
    if not patient_data[patient_data['nbpsys'].notna() & ((patient_data['nbpsys'] < healthy_nbp_sys_min) | (patient_data['nbpsys'] > healthy_nbp_sys_max))].empty:
        nbp_sys_flag = 1
    # New check for SpO2
    if not patient_data[patient_data['spo2'].notna() & (patient_data['spo2'] < spo2_threshold)].empty:
        spo2_flag = 1

    # Store flags for each patient
    patient_flags[patient_id] = {
        'hr_flag': hr_flag,
        'resp_flag': resp_flag,
        'nbp_sys_flag': nbp_sys_flag,
        'spo2_flag': spo2_flag
    }

# Create a list of features and labels
features = []
labels = []

for patient_id in train_demos_df['patient_id'].unique():
    flags = patient_flags.get(patient_id, {'hr_flag': 0, 'resp_flag': 0, 'nbp_sys_flag': 0, 'spo2_flag': 0})
    label = train_labels_df[train_labels_df['patient_id'] == patient_id]['label'].iloc[0]  # Ensure correct label column name
    features.append([flags['hr_flag'], flags['resp_flag'], flags['nbp_sys_flag'], flags['spo2_flag']])
    labels.append(label)

# Convert lists to NumPy arrays for modeling
X = np.array(features)
y = np.array(labels, dtype=int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auroc_score = roc_auc_score(y_test, y_pred_proba)

print("AUROC Score:", auroc_score)