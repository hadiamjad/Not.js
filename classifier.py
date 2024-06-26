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

def generateFeatureCDF(name, label ,xlim, output):
  # Load data
  df = pd.read_csv('data/notjs.csv')
  df = df.loc[:, ['script_name','method_name', 'label', name]]
  # Map the labels
  label_mapping = {0: 'non-tracking', 1: 'tracking'}
  df['label'] = df['label'].replace(label_mapping)

  # Create the CDF plot with y-axis ranging from 0 to 1
  kdep = sns.kdeplot(data=df, x=name, hue='label',
      cumulative=True, common_norm=False, common_grid=True)

  plt.xlim(0, xlim)
  plt.xlabel(label)
  plt.savefig('plots/'+ output + '.png')
  plt.show()

def generateSurrogateCDF(name1, name2, label, xlim, output):
  df = pd.read_csv("data/surrogate-generate-cdf.csv")

  # Melt the DataFrame to make it suitable for sns.ecdfplot
  df_melted = df.melt(id_vars=['website'], value_vars=[name1, name2])

  # Create the CDF plot
  kdep = sns.ecdfplot(data=df_melted, x='value', hue='variable')
  plt.xlabel(label)
  plt.ylabel('CDF')
  plt.xlim(0, xlim)
  # plt.title('CDF of functional requests')
  plt.savefig('plots/' + output + '_cdf.png')
  plt.show()

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

print(f'#################### Table 6 ####################')
# Load the JSON files
# Read values_output.json into a dictionary
with open('data/performance_output.json', 'r') as f:
    values_output = json.load(f)
# Read values_output_surr.json into a dictionary
with open('data/performance_output_surr.json', 'r') as f:
    values_output_surr = json.load(f)
# Function to remove outliers using the IQR method
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]
# Function to calculate mean, median, and mode
def calculate_statistics(data):
    if not data:
        return "No data"
    try:
        mean = np.mean(data)
        median = np.median(data)
        mode = statistics.mode(data)
        return mean, median, mode
    except statistics.StatisticsError:
        # If there's no unique mode
        return mean, median, "No unique mode"
# Compute and print the statistics for output after removing outliers
print("Statistics for Normal:")
for key in values_output:
    filtered_values_output = remove_outliers(values_output[key])
    stats = calculate_statistics(filtered_values_output)
    print(f"{key} - Mean: {stats[0]}, Median: {stats[1]}")
# Compute and print the statistics for output_surr after removing outliers
print("Statistics for Surrogate:")
for key in values_output_surr:
    filtered_values_output_surr = remove_outliers(values_output_surr[key])
    stats = calculate_statistics(filtered_values_output_surr)
    print(f"{key} - Mean: {stats[0]}, Median: {stats[1]}")

print('#################### Generating Plots ####################')
generateFeatureCDF('Feature 21', 'Number of Successor Function', 2000, 'Figure_7_a')
generateFeatureCDF('Feature 23', 'Number of Storage Access',200, 'Figure_7_b')

generateSurrogateCDF('normal_tracking functions', 'surr_tracking functions', 'Number of Tracking Functions', 5000, 'Figure_8_a')
generateSurrogateCDF('normal_tracking requests', 'surr_tracking requests', 'Number of Tracking Requests', 400, 'Figure_8_b')
generateSurrogateCDF('normal_functional requests', 'surr_functional requests', 'Number of Non-tracking Requests', 1000, 'Figure_8_c')

print('#################### All Plots Generated Successfully ####################')