from replaceFunctionCall import replace_function_call
import pandas as pd
import os
import json
import logging


def request_response_dic(filename):
    dataset = pd.read_json(filename, lines=True)
    request_response_dic = {}
    for i in dataset.index:
        if dataset["http_req"][i] not in request_response_dic:
            request_response_dic[dataset["http_req"][i]] = dataset["request_id"][i]
    return request_response_dic


def get_tracking_functions(filename):
    dataset = pd.read_excel(filename)
    tracking_functions_dic = {}
    for i in dataset.index:
        if dataset["label"][i] == 1:
            if dataset["script_name"][i] not in tracking_functions_dic:
                tracking_functions_dic[dataset["script_name"][i]] = []
            tracking_functions_dic[dataset["script_name"][i]].append(
                dataset["method_name"][i]
            )
    return tracking_functions_dic


def contains_only_numbers(input_string):
    # Remove leading and trailing whitespaces
    input_string = input_string.strip()

    # Check if the string contains only digits after removing a possible decimal point
    return input_string.replace(".", "", 1).isdigit()


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",
    )


def main():
    fold = os.listdir("server/output")
    folder = "server/output/"
    # save logs
    log_file = "logs/function_logs.json"
    setup_logging(log_file)

    for f in fold:
        try:
            print("Starting surrogate-generation: website: ", f)
            # main stuff
            logging.info("generating-surrogates: %s", f)
            # {request_url: request_id}
            request_id = request_response_dic(folder + f + "/request.json")
            # {script_name/script_url: [method_name@line_number@column_number, ...]}
            tracking_functions = get_tracking_functions(folder + f + "/features.xlsx")
            # logs
            logs = {
                "script_not_in_request_file": 0,
                "inline_script": 0,
                "replace_function_call_fail": 0,
                "success": 0,
            }

            for script_name in tracking_functions:
                # If the script's request_id not found in the dataset
                if script_name not in request_id:
                    for method in tracking_functions[script_name]:
                        logs["script_not_in_request_file"] += 1
                        logging.info(
                            "script_not_in_request_file %s %s", script_name, method
                        )
                else:
                    req_id = request_id[script_name]
                    if not contains_only_numbers(req_id):
                        logging.info(
                            "inline_script %s %s",
                            script_name,
                            tracking_functions[script_name],
                        )
                        # print(f"Inline script {req_id} for {script_name}")
                        logs["inline_script"] += len(tracking_functions[script_name])
                    else:
                        for method in tracking_functions[script_name]:
                            line_num = int(method.split("@")[1]) + 1
                            column_num = int(method.split("@")[2]) + 1
                            try:
                                logging.info(
                                    "Replacing function call at line %s column %s for %s and request_id %s",
                                    line_num,
                                    column_num,
                                    script_name,
                                    req_id,
                                )
                                status = replace_function_call(
                                    folder + f + "/response/" + req_id + ".txt",
                                    folder
                                    + f
                                    + "/surrogate/"
                                    + req_id
                                    + "_modified.txt",
                                    line_num,
                                    column_num,
                                )
                                if status == 0:
                                    logs["success"] += 1
                                else:
                                    logs["replace_function_call_fail"] += 1
                                    logging.info(
                                        "Crashed replacing function call at line %s column %s for %s and error end index not found",
                                        line_num,
                                        column_num,
                                        script_name,
                                    )
                            except Exception as e:
                                logging.info(
                                    "Crashed replacing function call at line %s column %s for %s and error %s",
                                    line_num,
                                    column_num,
                                    script_name,
                                    e,
                                )
                                logs["replace_function_call_fail"] += 1
            json.dump(request_id, open(folder + f + "/request_id.json", "w"))
            json.dump(logs, open(folder + f + "/surrogate_logs.json", "w"))
            logging.info("Total Logs %s", logs)
            print("Completed surrogate-generation: website: ", f)
        except Exception as e:
            print("Crashed surrogate-generation: website: ", f, e)


if __name__ == "__main__":
    main()
