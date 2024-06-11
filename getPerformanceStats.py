import os
import json
import numpy as np
import statistics

# Function to remove outliers using the IQR method
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

# Initialize dictionaries to store all values for each metric
values_output = {"dom_content_loaded": [], "dom_interactive": [], "load_event_time": [], "total_heap_size": [], "used_heap_size": []}
values_output_surr = {"dom_content_loaded": [], "dom_interactive": [], "load_event_time": [], "total_heap_size": [], "used_heap_size": []}

# Get the list of folders
folders = os.listdir("server/output")

for folder in folders:
    # Check and read performance.json in output
    output_path = os.path.join("server/output", folder, "performance.json")
    # Check and read performance.json in output_surr
    output_surr_path = os.path.join("server/output_surr", folder, "performance.json")
    if os.path.isfile(output_path) and os.path.isfile(output_surr_path):
        with open(output_path) as file:
            data = json.load(file)
            for key in values_output:
                if key in data:
                    values_output[key].append(data[key])

        with open(output_surr_path) as file:
            data = json.load(file)
            for key in values_output_surr:
                if key in data:
                    values_output_surr[key].append(data[key])

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
print("Statistics for output (Outliers Removed):")
for key in values_output:
    filtered_values = remove_outliers(values_output[key])
    stats = calculate_statistics(filtered_values)
    print(f"{key} - Mean: {stats[0]}, Median: {stats[1]}, Mode: {stats[2]}")

# Compute and print the statistics for output_surr after removing outliers
print("Statistics for output_surr (Outliers Removed):")
for key in values_output_surr:
    filtered_values = remove_outliers(values_output_surr[key])
    stats = calculate_statistics(filtered_values)
    print(f"{key} - Mean: {stats[0]}, Median: {stats[1]}, Mode: {stats[2]}")
