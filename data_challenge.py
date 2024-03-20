import pandas as pd

# Load the training demographics data
train_demos_df = pd.read_csv("train/train_demos.csv")

# Display the first few rows of the demographics data
print("Demographics Data:")
print(train_demos_df.head())

# Extract demographic features
demographic_features = train_demos_df[["age", "gender", "ethnicity", "marital_status"]]

# Perform one-hot encoding for categorical variables (gender, ethnicity, marital_status)
demographic_features = pd.get_dummies(demographic_features, columns=["gender", "ethnicity", "marital_status"])

# Display the extracted demographic features
print("\nExtracted Demographic Features:")
print(demographic_features.head())
