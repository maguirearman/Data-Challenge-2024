import pandas as pd
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

# Define healthy range for heart rate
healthy_min = 60
healthy_max = 100


# Get all unique patient IDs from train_demos
all_patient_ids = train_demos_df['patient_id'].unique()

# Initialize feature vector with all patient IDs and set flags to None initially
feature_vector = pd.DataFrame({'patient_id': all_patient_ids, 'flag': None})

# Iterate through each patient's heart rate measurements
for patient_id, patient_data in train_signs_df.groupby('patient_id'):
    # Check if there are any heart rate measurements for the patient
    if 'heartrate' in patient_data.columns:
        # Check if any heart rate measurement is outside the healthy range
        if (patient_data['heartrate'] < healthy_min).any() or (patient_data['heartrate'] > healthy_max).any():
            feature_vector.loc[feature_vector['patient_id'] == patient_id, 'flag'] = 1  # If any heart rate measurement is outside the healthy range, set flag to 1
        else:
            feature_vector.loc[feature_vector['patient_id'] == patient_id, 'flag'] = 0  # Otherwise, set flag to 0



# Merge feature vector with labels
feature_vector_with_labels = pd.merge(feature_vector, train_labels_df, on='patient_id')


print("Feature vector:")
print(feature_vector_with_labels.head())    

# Prepare features (flag) and labels
X = feature_vector_with_labels[['flag']].values
y = feature_vector_with_labels['label'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUROC score
auroc_score = roc_auc_score(y_test, y_pred_proba)

print("AUROC Score:", auroc_score)