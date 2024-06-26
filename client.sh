#!/bin/bash
# Path to the CSV file containing the values
CSV_FILE="test.csv"
# Check if the CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi
# Read the CSV file and loop over each line
while IFS= read -r line; do
    # Run the Python scripts with the current value as an argument
    python3 sele.py "$line"
    python3 sele-hook.py "$line"
done < "$CSV_FILE"
python3 -W ignore label.py 
python3 -W ignore graph-plot/main.py
python3 -W ignore graph-plot/makeFeatures.py
python3 -W ignore surrogate/main.py
python3 -W ignore generateSurrogateForChrome.py

