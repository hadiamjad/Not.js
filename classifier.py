import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import json
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print(f'#################### Downloading Dataset and Models ####################')
# # # URL of your .joblib file on Google Drive
notjs = 'https://drive.google.com/uc?id=14jwOXwHjGTJnaIO80Lj9d7ZAZHLgXKl1'
notjs_data = 'https://drive.google.com/uc?id=18uuLj8OGxUwIRAXOK154GwvgRDG5Fj5t'
notjs_ablation_1 = 'https://drive.google.com/uc?id=1LUNw06sy-LwKQMpSvIFYR-KUMTgEEMyU'
notjs_ablation_2 = 'https://drive.google.com/uc?id=10QZg9O1Xx41jHHnQain-3TX7gVIbToom'
performance_output = 'https://drive.google.com/uc?id=1LH6b1z2Pb6XlpbrQ2_QRJu1X89MAQk-U'
performance_output_surr = 'https://drive.google.com/uc?id=1dT7pjKmq_e0yAieSN5aSn-Eep78p3y24'
surrogate_generate_cdf = 'https://drive.google.com/uc?id=1nIGfuabPrumrf_eQc5qLEsVJ8hRXwdBo'
test_coverage = 'https://drive.google.com/uc?id=1msw31VsfjVj51828VZeHdzaJU45j4sED'
test_obfuscation = 'https://drive.google.com/uc?id=1ULT44VnvP0ZXT6sJ4Mtal-RCfgkk6nxc'

# # Download the file
gdown.download(notjs, 'data/notjs.joblib', quiet=False)
gdown.download(notjs_data, 'data/notjs.csv', quiet=False)
gdown.download(notjs_ablation_1, 'data/notjs_ablation_1.joblib', quiet=False)
gdown.download(notjs_ablation_2, 'data/notjs_ablation_2.joblib', quiet=False)
gdown.download(performance_output, 'data/performance_output.json', quiet=False)
gdown.download(performance_output_surr, 'data/performance_output_surr.json', quiet=False)
gdown.download(surrogate_generate_cdf, 'data/surrogate-generate-cdf.csv', quiet=False)
gdown.download(test_coverage, 'data/test_coverage.csv', quiet=False)
gdown.download(test_obfuscation, 'data/test_obfuscation.csv', quiet=False)

np.random.seed(42)

def getResults(path):
  df = pd.read_csv(path)

  df = df[df['num_req_sent'] != 0]
  # Assuming you have a DataFrame 'final_df' with 'label' and 'prediction' columns
  fn = ((df['label'] == 1) &  (df['prediction'] == 0)).sum()
  tp = ((df['label'] == 1) &  (df['prediction'] == 1)).sum()
  # Assuming you have a DataFrame 'final_df' with 'label' and 'prediction' columns
  tn = ((df['label'] == 0) &  (df['prediction'] == 0)).sum()
  fp= ((df['label'] == 0) &  (df['prediction'] == 1)).sum()
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1score = (2 * precision * recall )/(precision + recall)
  print(f'Precision: {precision:.2f}')
  print(f'Recall: {recall:.2f}')
  print(f'F1-score: {f1score:.2f}')

def read_dataset(file):
  df = pd.read_csv(file)
  df = df.drop_duplicates(
    subset = ["script_name","method_name"],
    keep = 'last').reset_index(drop = True)
  return df

def generateConfig(df, lst, is_init, start):
  if is_init:
    df = df[df['Feature 25'] == 1]
  weights = df['Feature 2'].values
  df = df.drop(lst, axis=1)


  labels = df['label'].values
  features  = df.iloc[:, start:].values

  return labels, features, weights

dataset = read_dataset(r'data/notjs.csv')
labels, features, weights = generateConfig(dataset, ['Feature 2'], False, 4)
X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(features, labels, weights, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val, sample_weights_train, sample_weights_val = train_test_split(X_train, y_train, sample_weights_train, test_size=0.25, random_state=42)
print('test-training split done')

## Generate values for table 3 in the paper
print(f'#################### Table 3 ####################')
# Gather shapes
shapes = {
    'train': X_train.shape,
    'val': X_val.shape,
    'test': X_test.shape
}
# Gather class counts
class_counts_train = np.bincount(y_train)
class_counts_val = np.bincount(y_val)
class_counts_test = np.bincount(y_test)
# Create a DataFrame
data = {
    'Tracking Functions': [class_counts_train[1], class_counts_val[1], class_counts_test[1]],
    'Non-tracking Functions': [class_counts_train[0], class_counts_val[0], class_counts_test[0]],
    'Total': [class_counts_train.sum(), class_counts_val.sum(), class_counts_test.sum()],
}
df = pd.DataFrame(data, index=['Training', 'Validation', 'Testing'])
# Print the table
print(df)

# Load the model
model = load('data/notjs.joblib')
print('#################### Model Loaded Successfully ####################')

# Make predictions
y_pred = model.predict(X_test)

# # Compute precision and recall
# Compute the accuracy of the model on the testing data
report = classification_report(y_test, y_pred, sample_weight=sample_weights_test, output_dict=True)
print(f'#################### Table 4 - Standard 5.1 ####################')
# Extracting precision, recall, and f1-score for class label 1
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']

# Printing the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

print(f'#################### Table 4 - Obfuscation 5.2 ####################')
getResults(r'data/test_obfuscation.csv')

print(f'#################### Table 4 - Coverage 5.3 ####################')
getResults(r'data/test_coverage.csv')

print(f'#################### Table 5 - Row 1 ####################')
dataset = read_dataset(r'data/notjs.csv')
lst = [
    "Feature 15", "Feature 1", "Feature 21", "Feature 26", 
    "Feature 23", "Feature 24", "Feature 37", "Feature 38", 
    "Feature 39", "Feature 22", "Feature 25", "Feature 31", 
    "Feature 32", "Feature 33", "Feature 34", "Feature 35", 
    "Feature 36", "Feature 40", "Feature 41", "Feature 42", 
    "Feature 43"
]
labels, features, weights = generateConfig(dataset, lst, True, 5)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
# Load the model
model = load('data/notjs_ablation_1.joblib')
# Make predictions
y_pred = model.predict(X_test)
# # Compute precision and recall
# Compute the accuracy of the model on the testing data
report = classification_report(y_test, y_pred, output_dict=True)
# Extracting precision, recall, and f1-score for class label 1
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']
# Printing the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

print(f'#################### Table 5 - Row 2 ####################')
dataset = read_dataset(r'data/notjs.csv')
lst = [
    "Feature 15", "Feature 1", "Feature 21", "Feature 26", 
    "Feature 23", "Feature 24", "Feature 37", "Feature 38", 
    "Feature 39", "Feature 22", "Feature 25", "Feature 31", 
    "Feature 32", "Feature 33", "Feature 34", "Feature 35", 
    "Feature 36", "Feature 40", "Feature 41", "Feature 42", 
    "Feature 43"
]
labels, features, weights = generateConfig(dataset, lst, False, 5)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
# Load the model
model = load('data/notjs_ablation_2.joblib')
# Make predictions
y_pred = model.predict(X_test)
# # Compute precision and recall
# Compute the accuracy of the model on the testing data
report = classification_report(y_test, y_pred, output_dict=True)
# Extracting precision, recall, and f1-score for class label 1
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']
# Printing the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

print('#################### Generating Plots ####################')
