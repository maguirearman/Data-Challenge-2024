import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
aniongap_normal_min, aniongap_normal_max = 3, 12
bicarbonate_normal_min, bicarbonate_normal_max = 22, 29
chloride_normal_min, chloride_normal_max = 96, 107
creatine_normal_min, creatine_normal_max = 0.6, 1.3
calcium_normal_min, calcium_normal_max = 8.5, 10.4
glucose_normal_min, glucose_normal_max = 100, 126
hematocrit_normal_min, hematocrit_normal_max = 38.3, 50.0
hemoglobin_normal_min, hemoglobin_normal_max = 12.0, 17.5
mch_normal_min, mch_normal_max = 27, 31
mcv_normal_min, mcv_normal_max = 80, 100
magnesium_normal_min, magnesium_normal_max = 0.5, 3.0
phosphate_normal_min, phosphate_normal_max = 2.5, 4.5
platelet_normal_min, platelet_normal_max = 150, 450
potassium_normal_min, potassium_normal_max = 3.0, 6.0

# Process training data to calculate flags based on thresholds
flags_df = train_signs_df.groupby('patient_id').agg({
    'heartrate': lambda hr: ((hr < healthy_hr_min) | (hr > healthy_hr_max)).any().astype(int),
    'resp': lambda r: ((r < healthy_resp_min) | (r > healthy_resp_max)).any().astype(int),
    'nbpsys': lambda nbp: ((nbp < healthy_nbp_sys_min) | (nbp > healthy_nbp_sys_max)).any().astype(int),
    'spo2': lambda s: (s < spo2_threshold).any().astype(int),
    'nbpdia': lambda dia: ((dia < healthy_nbpdia_min) | (dia > healthy_nbpdia_max)).any().astype(int), 
    'nbpmean': lambda mean: ((mean < nbpmean_threshold_min) | (mean > nbpmean_threshold_max)).any().astype(int),
    'aniongap': lambda ag: ((ag < aniongap_normal_min) | (ag > aniongap_normal_max)).any().astype(int) ,
    'bicarbonate': lambda bic: ((bic < bicarbonate_normal_min) | (bic > bicarbonate_normal_max)).any().astype(int),
    'chloride': lambda cl: ((cl < chloride_normal_min) | (cl > chloride_normal_max)).any().astype(int),
    'creatinine': lambda cr: ((cr < creatine_normal_min) | (cr > creatine_normal_max)).any().astype(int),
    'calcium': lambda ca: ((ca < calcium_normal_min) | (ca > calcium_normal_max)).any().astype(int),
    'glucose': lambda glu: ((glu < glucose_normal_min) | (glu > glucose_normal_max)).any().astype(int),
    'hematocrit': lambda hct: ((hct < hematocrit_normal_min) | (hct > hematocrit_normal_max)).any().astype(int),
    'hemoglobin': lambda hgb: ((hgb < hemoglobin_normal_min) | (hgb > hemoglobin_normal_max)).any().astype(int),
    'mch': lambda mch: ((mch < mch_normal_min) | (mch > mch_normal_max)).any().astype(int),
    'mcv': lambda mcv: ((mcv < mcv_normal_min) | (mcv > mcv_normal_max)).any().astype(int),
    'magnesium': lambda mg: ((mg < magnesium_normal_min) | (mg > magnesium_normal_max)).any().astype(int),
    'phosphate': lambda phos: ((phos < phosphate_normal_min) | (phos > phosphate_normal_max)).any().astype(int),
    'platelet': lambda plt: ((plt < platelet_normal_min) | (plt > platelet_normal_max)).any().astype(int),
    'potassium': lambda k: ((k < potassium_normal_min) | (k > potassium_normal_max)).any().astype(int)

}).reset_index()

# Merge flags with labels
features_df = pd.merge(flags_df, train_labels_df, on='patient_id', how='left')

# Calculate counts for each vital measure
vital_counts_df = train_signs_df.groupby('patient_id').agg({
    'heartrate': 'count',
    'resp': 'count',
    'nbpsys': 'count',
    'spo2': 'count',
    'nbpdia': 'count',
    'nbpmean': 'count',
    'aniongap': 'count',
    'bicarbonate': 'count',
    'chloride': 'count',
    'creatinine': 'count',
    'calcium': 'count',
    'glucose': 'count',
    'hematocrit': 'count',
    'hemoglobin': 'count',
    'mch': 'count',
    'mcv': 'count',
    'magnesium': 'count',
    'phosphate': 'count',
    'platelet': 'count',
    'potassium': 'count'
}).reset_index()

# Sum the counts to get a total measure count per patient
vital_counts_df['total_vital_measures'] = vital_counts_df.drop('patient_id', axis=1).sum(axis=1)

# Keep only patient_id and total_vital_measures for merging
vital_measures_total_df = vital_counts_df[['patient_id', 'total_vital_measures']]

# Merge with the features DataFrame
features_df = pd.merge(features_df, vital_measures_total_df, on='patient_id', how='left')

# Preprocessing demographic data

# Convert 'admittime' to datetime
train_demos_df['admittime'] = pd.to_datetime(train_demos_df['admittime'])

# Extract useful features from 'admittime' (e.g., hour of admission, month)
train_demos_df['admit_hour'] = train_demos_df['admittime'].dt.hour
train_demos_df['admit_month'] = train_demos_df['admittime'].dt.month

# Convert 'gender' into a binary variable
train_demos_df['gender'] = train_demos_df['gender'].map({'M': 1, 'F': 0})

# One-hot encode categorical variables ('insurance', 'marital_status', 'ethnicity')
train_demos_df = pd.get_dummies(train_demos_df, columns=['insurance', 'marital_status', 'ethnicity'], drop_first=True)

# Pre process radiology data
# Done in radio.py
# Load the radiology feature vector CSV
radiology_features_df = pd.read_csv('radiology_feature_vector.csv')

# Set k to the number of features to keep
k=500
# Merge radio data vector as well

features_df = pd.merge(features_df, radiology_features_df, on='patient_id', how='left')
radiology_feature_columns = [str(i) for i in range(0, k)]  # Adjust this range based on your feature columns

features_df = pd.merge(features_df, train_demos_df.drop(['admittime'], axis=1), on='patient_id', how='left')

# Prepare features and labels for modeling
X_columns = ['heartrate', 'resp', 'nbpsys', 'spo2', 'nbpdia', 'nbpmean', 'aniongap', 'bicarbonate', 'chloride', 'creatinine', 'calcium', 'glucose', 'hematocrit', 'hemoglobin', 'mch', 'magnesium', 'phosphate', 'platelet', 'potassium', 'age', 'gender', 'admit_hour', 'admit_month', 'total_vital_measures'] + \
            [col for col in features_df.columns if col.startswith('insurance_') or col.startswith('marital_status_') or col.startswith('ethnicity_')]  + \
            radiology_feature_columns

X = features_df[X_columns].to_numpy()
y = features_df['label'].to_numpy().astype(int)

# Split the data for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence

# Train the model
log_reg_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_val_pred_proba = log_reg_model.predict_proba(X_val_scaled)[:, 1]
val_auroc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Logistic Regression Validation AUROC Score: {val_auroc_score:.4f}")


# Prepare a pipeline for scaling and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
])

# Parameters grid for Logistic Regression
param_grid = {
    'log_reg__C': np.logspace(-4, 4, 20),
    'log_reg__solver': ['liblinear']  # 'liblinear' is a good choice for small datasets and binary classification
}

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Evaluate the best model from grid search
best_model = grid_search.best_estimator_
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
val_auroc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Logistic Regression Best Model (with GridSearch) Validation AUROC Score: {val_auroc_score:.4f}")

# Define other models for comparison
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GBM (Gradient Boosting)': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Iterate over models, train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Assuming scaling is beneficial
    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # Probability estimates
    val_auroc_score = roc_auc_score(y_val, y_val_pred_proba)
    print(f"{name} Validation AUROC Score: {val_auroc_score:.4f}")