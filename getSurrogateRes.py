import os
import json

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
count = 0
surr_average = 0
norm_average = 0
surr_tracking = 0
norm_tracking = 0
surr_functional = 0
norm_functional = 0
script_not_in_request_file = 0
inline_script = 0
replace_function_call_fail = 0
replace_function_call_success = 0

storage_access_surr = 0
storage_access_norm = 0

for f in fold:
    try:
        if len(os.listdir("server/output_surr/" + f + "/surrogate")) > 0:
            surr_gen = {"script_not_in_request_file": 0, "inline_script": 0, "replace_function_call_fail": 0, "success": 0}
            with open("server/output_surr/" + f + "/surrogate_logs.json") as file:
                # Load JSON data from file
                data = json.load(file)
                surr_gen["script_not_in_request_file"] = data["script_not_in_request_file"]
                surr_gen["inline_script"] = data["inline_script"]
                surr_gen["replace_function_call_fail"] = data["replace_function_call_fail"]
                surr_gen["success"] += data["success"]
                # print("Surrogate_logs", data)
            try:
                with open("server/output_surr/" + f + "/cookie_storage.json", 'r') as file:
                    lines = file.readlines()
                    storage_access_norm += len(lines)
                with open("server/output/" + f + "/cookie_storage.json", 'r') as file:
                    lines = file.readlines()
                    storage_access_surr += len(lines)
            except:
                pass

            surr = {"tracking-functions": [], "inline-functions": [], "tracking-requests": 0, "functional-requests":0 }    
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
                            surr["tracking-requests"] += 1
                            lst = CheckAncestoralNodes(dataset["call_stack"])
                            for item in lst:
                                if item not in surr["tracking-functions"] and "https://" + dataset["top_level_url"] + "/" != item:
                                    surr["tracking-functions"].append(item)
                                elif item not in surr["inline-functions"] and   "https://" + dataset["top_level_url"] + "/" == item:
                                    surr["inline-functions"].append(item)
                        else:
                            surr["functional-requests"] += 1
            # print("Surrogate",f,{k: len(v) if isinstance(v, list) else v for k, v in surr.items()})

            norm = {"tracking-functions": [], "inline-functions": [], "tracking-requests": 0, "functional-requests":0 }
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
                            norm["tracking-requests"] += 1
                            lst = CheckAncestoralNodes(dataset["call_stack"])
                            for item in lst:
                                if item not in norm["tracking-functions"] and "https://" + dataset["top_level_url"] + "/" != item.split("@")[0]:
                                    norm["tracking-functions"].append(item)
                                elif item not in norm["inline-functions"] and   "https://" + dataset["top_level_url"] + "/" == item.split("@")[0]:
                                    norm["inline-functions"].append(item)
                        else:
                            norm["functional-requests"] += 1
            # if surr["tracking-functions"] < norm["tracking-functions"]:
            count += 1
            surr_average += len(surr["tracking-functions"])
            norm_average += len(norm["tracking-functions"])
            surr_tracking += surr["tracking-requests"]
            norm_tracking += norm["tracking-requests"]
            surr_functional += surr["functional-requests"]
            norm_functional += norm["functional-requests"]
            script_not_in_request_file += surr_gen["script_not_in_request_file"]
            inline_script += surr_gen["inline_script"]
            replace_function_call_fail += surr_gen["replace_function_call_fail"]
            replace_function_call_success += surr_gen["success"]

            # print("Before-surrogate",f,{k: len(v) if isinstance(v, list) else v for k, v in norm.items()})
                    
    except:
        pass
print("total-websites", count)
print("Average tracking requests after surrogate/website", (surr_tracking))
print("Average tracking requets before surrogate/website", (norm_tracking))
print("Average functional requests after surrogate/website", (surr_functional))
print("Average functional requests before surrogate/website", (norm_functional))
print("Average tracking functions after surrogate/website", (surr_average))
print("Average tracking functions before surrogate/website", (norm_average))
print("Average script_not_in_request_file", (script_not_in_request_file))
print("Average inline_script", (inline_script))
print("Average replace_function_call_fail", (replace_function_call_fail))
print("Average replace_function_call_success", (replace_function_call_success))
print("Average storage access after surrogate", (storage_access_surr))
print("Average storage access before surrogate", storage_access_norm)