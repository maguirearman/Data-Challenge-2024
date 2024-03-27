import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

# Preprocessing vitals data

# Define thresholds
healthy_hr_min, healthy_hr_max = 60, 100
healthy_resp_min, healthy_resp_max = 12, 25
healthy_nbp_sys_min, healthy_nbp_sys_max = 90, 140
spo2_threshold = 90
healthy_nbpdia_min, healthy_nbpdia_max = 70, 90
nbpmean_threshold_min, nbpmean_threshold_max = 70, 95 

# Process training data to calculate flags
flags_df = train_signs_df.groupby('patient_id').agg({
    'heartrate': lambda hr: ((hr < healthy_hr_min) | (hr > healthy_hr_max)).any().astype(int),
    'resp': lambda r: ((r < healthy_resp_min) | (r > healthy_resp_max)).any().astype(int),
    'nbpsys': lambda nbp: ((nbp < healthy_nbp_sys_min) | (nbp > healthy_nbp_sys_max)).any().astype(int),
    'spo2': lambda s: (s < spo2_threshold).any().astype(int),
    'nbpdia': lambda dia: ((dia < healthy_nbpdia_min) | (dia > healthy_nbpdia_max)).any().astype(int), 
    'nbpmean': lambda mean: ((mean < nbpmean_threshold_min) | (mean > nbpmean_threshold_max)).any().astype(int) 
}).reset_index()

# Merge flags with labels
features_df = pd.merge(flags_df, train_labels_df, on='patient_id', how='left')

# Preprocessing demographic data

# Convert 'admittime' to datetime
train_demos_df['admittime'] = pd.to_datetime(train_demos_df['admittime'])

# Extract useful features from 'admittime' (e.g., hour of admission, month)
train_demos_df['admit_hour'] = train_demos_df['admittime'].dt.hour
train_demos_df['admit_month'] = train_demos_df['admittime'].dt.month

# Convert 'gender' into a binary variable
train_demos_df['gender'] = train_demos_df['gender'].map({'M': 1, 'F': 0})

# One-hot encode categorical variables ('insurance', 'marital_status', 'ethnicity')
# Note: Consider filling missing values if any
train_demos_df = pd.get_dummies(train_demos_df, columns=['insurance', 'marital_status', 'ethnicity'], drop_first=True)

# Merge demographic features with clinical flags
features_df = pd.merge(features_df, train_demos_df.drop(['admittime'], axis=1), on='patient_id', how='left')

# Assuming 'label' column exists in 'train_labels_df' and is correctly named
# Prepare features and labels for modeling
X_columns = ['heartrate', 'resp', 'nbpsys', 'spo2', 'nbpdia', 'nbpmean', 'age', 'gender', 'admit_hour', 'admit_month'] + \
            [col for col in features_df.columns if col.startswith('insurance_') or col.startswith('marital_status_') or col.startswith('ethnicity_')]
X = features_df[X_columns].to_numpy()
y = features_df['label'].to_numpy().astype(int)

# Split the data for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Dictionary of models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GBM (XGBoost)": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Iterate over models, train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability estimates
    val_auroc_score = roc_auc_score(y_val, y_val_pred_proba)
    print(f"{name} Validation AUROC Score: {val_auroc_score:.4f}")