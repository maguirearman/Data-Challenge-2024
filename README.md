# In-hospital Mortality Prediction on ICU Data

## Introduction

![](./data_.png)

- Goal: In-hospital mortality using demographics and first 24 hours of ICU data after admission
- Evaluation: AUROC
  - Baseline by TA: 0.785
  - Submission: CSV file with `patient_id, probability` as headers

## Data

### Data Description

- train (13,708 patients, 534,422 records, 27,922 radiology notes)
  - train_labels.csv: training labels
  - train_demos.csv: training demographics
  - train_signs.csv: first 24 hours ICU data extracted, including vital signs and lab tests
  - train_radiology.csv: training radiology notes, within the first 24 hours
- test (3,427 patients, 132,608 records, 7,123 radiology notes)
  - test_labels.csv: test labels (hidden)
  - test_demos.csv: test demographics
  - train_signs.csv: first 24 hours ICU data extracted, including vital signs and lab tests
  - train_radiology.csv: test radiology notes, within the first 24 hours

### Data Remarks

- In each file, `patient_id` is the de-identified patient ID.
- In the label file, `1` on the `mortality` column means in-hospital mortality while `0` means discharge.
- In the signs file, `charttime` is the time that features are collected.
- In the radiology file, there are two types of notes denoted by `note_type` field: RR (radiology report) and AR(radiology report addendum).
- In the radiology file, multiple notes might occur at the same chart time. Use `note_seq` to differentiate them.

### Submission

- AUROC is used as the evaluation metric.
- CSV file with `patient_id, probability` as headers, each line for patient ID and the probability of the patient's in-hospital mortality.
- Please make sure the order of patients' IDs is the same as those in the label file.

### Files in Directory

- data_challenge.py: Main program where feature generation, model training, and output generation takes place
- radio.py: Side program where radiology_feature_vectore.csv is generated
- test_radio.py: Side program where test_radiology_feature_vectore.csv is generated 
- medical_terms_and_defs.txt: The medical dictionary we used
- submission.csv: Output csv file

