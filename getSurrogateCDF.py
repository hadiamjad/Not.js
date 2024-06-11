import os
import json
import pandas as pd

# these two functions and implementation is borrowed from label.py ancestor labelling
def CheckAncestoralNodes(callstack):
    # Handling non-script type
    if callstack["type"] != "script":
        return []

    # Initialize a set to track unique script URLs
    unique_scripts = set()
    
    # Recursively insert unique scripts in the stack
    rec_stack_checker(callstack["stack"], unique_scripts)

    # Convert the set to a list and return it
    return list(unique_scripts)

def rec_stack_checker(stack, unique_scripts):
    # Append unique script URLs to the set
    for item in stack["callFrames"]:
        script_url = (
            item["url"]
            + "@"
            + item["functionName"]
            + "@"
            + str(item["lineNumber"])
            + "@"
            + str(item["columnNumber"])
        )
        unique_scripts.add(script_url)

    # Check if the parent object exists and send a recursive call
    if "parent" in stack:
        rec_stack_checker(stack["parent"], unique_scripts)

fold = os.listdir("server/output")

website = {}

for f in fold:
    try:
        if len(os.listdir("server/output_surr/" + f + "/surrogate")) > 0:
            website[f] = {"surr":{"tracking-functions": 0, "inline-functions": 0, "tracking-requests": 0, "functional-requests":0 }, 
                          "norm": {"tracking-functions": 0, "inline-functions": 0, "tracking-requests": 0, "functional-requests":0 }}    
            # reading big request data line by line
            with open("server/output/" + f + "/label_request.json") as file:
                for line in file:
                    data = json.loads(line)
                    for dataset in data:
                        if (
                            dataset["easylistflag"] == 1
                            or dataset["easyprivacylistflag"] == 1
                            or dataset["ancestorflag"] == 1
                        ) and dataset["call_stack"]["type"] == "script":
                            website[f]["surr"]["tracking-requests"] += 1
                            lst = CheckAncestoralNodes(dataset["call_stack"])
                            for item in lst:
                                website[f]["surr"]["tracking-functions"] += 1
                        else:
                            website[f]["surr"]["functional-requests"] += 1

            # reading big request data line by line
            with open("server/output_surr/" + f + "/label_request.json") as file:
                for line in file:
                    data = json.loads(line)
                    for dataset in data:
                        if (
                            dataset["easylistflag"] == 1
                            or dataset["easyprivacylistflag"] == 1
                            or dataset["ancestorflag"] == 1
                        )and dataset["call_stack"]["type"] == "script":    
                            website[f]["norm"]["tracking-requests"] += 1
                            lst = CheckAncestoralNodes(dataset["call_stack"])
                            for item in lst:
                                website[f]["norm"]["tracking-functions"]+= 1
                        else:
                            website[f]["norm"]["functional-requests"] += 1
                    
    except:
        pass
print("total-websites", len(website))
# Data list to hold the records for the DataFrame
df_data = []

# Loop through the dictionary to gather data
for site, metrics in website.items():
    row = {'website': site}
    row['normal_tracking functions'] = metrics['norm'].get('tracking-functions', None)
    row['surr_tracking functions'] = metrics['surr'].get('tracking-functions', None)
    row['normal_tracking requests'] = metrics['norm'].get('tracking-requests', None)
    row['surr_tracking requests'] = metrics['surr'].get('tracking-requests', None)
    row['normal_functional requests'] = metrics['norm'].get('functional-requests', None)
    row['surr_functional requests'] = metrics['surr'].get('functional-requests', None)
    df_data.append(row)

# Create the DataFrame
df = pd.DataFrame(df_data)

# Show the DataFrame
print(df)

df.to_csv("surrogate-generate-cdf.csv")

# # Melt the DataFrame to make it suitable for sns.ecdfplot
# df_melted = df.melt(id_vars=['website'], value_vars=['normal_functional requests', 'surr_functional requests'])

# # Create the CDF plot
# sns.ecdfplot(data=df_melted, x='value', hue='variable')
# plt.xlabel('functional requests')
# plt.ylabel('CDF')
# plt.title('CDF of functional requests')
# plt.show()
# plt.savefig('surrogate_functional_requests_cdf.png')