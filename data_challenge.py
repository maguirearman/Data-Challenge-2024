import pandas as pd

# Load the training demographics data
train_demos_df = pd.read_csv("train/train_demos.csv")

# Display the first few rows of the demographics data
# print("Demographics Data:")
# print(train_demos_df.head())

# Extract demographic features
demographic_features = train_demos_df[["age", "gender", "ethnicity", "marital_status"]]

# Perform one-hot encoding for categorical variables (gender, ethnicity, marital_status)
demographic_features = pd.get_dummies(demographic_features, columns=["gender", "ethnicity", "marital_status"])

# Display the extracted demographic features
# print("\nExtracted Demographic Features:")
# print(demographic_features.head())

train_signs_df = pd.read_csv("train/train_signs.csv")

# Define healthy range for heart rate
healthy_min = 60
healthy_max = 100

# Create a dictionary to store patient IDs and their corresponding flag
patient_flags = {}

# Iterate through each patient's heart rate measurements
for patient_id, patient_data in train_signs_df.groupby('patient_id'):
    # Check if any heart rate measurement is outside the healthy range
    if (patient_data['heartrate'] < healthy_min).any() or (patient_data['heartrate'] > healthy_max).any():
        # If any heart rate measurement is outside the healthy range, set flag to 1
        patient_flags[patient_id] = 1
    else:
        # Otherwise, set flag to 0
        patient_flags[patient_id] = 0

# Create a DataFrame from the patient flags dictionary
feature_vector = pd.DataFrame({'patient_id': list(patient_flags.keys()), 'flag': list(patient_flags.values())})

print("Feature vector:")
print(feature_vector)