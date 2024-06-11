## This file is used to generate surrogates for chrome 
## run either sele_surro.py or this file
import pandas as pd
import os
import json
import shutil


def request_response_dic(filename):
    dataset = pd.read_json(filename, lines=True)
    request_response_dic = {}
    for i in dataset.index:
        if dataset["request_id"][i] not in request_response_dic:
            request_response_dic[dataset["request_id"][i]] = dataset["http_req"][i]
    return request_response_dic

def create_directory_from_url(url, folder_path, file_path):
    # Remove the scheme (http:// or https://) from the URL
    url = url.replace('http://', '').replace('https://', '')

    # Extract the domain and path
    domain_and_path = url.split('/', 1)
    domain = domain_and_path[0]
    path = domain_and_path[1] if len(domain_and_path) > 1 else ''

    # Create the full directory path
    full_path = os.path.join(folder_path, domain, path)

    # Extract the directory and file name
    directory, file_name = os.path.split(full_path)

    # Create the directory structure
    os.makedirs(directory, exist_ok=True)

    # Copy the file to the target location
    shutil.copy(file_path, os.path.join(directory, file_name))

def main():
    fold = os.listdir("server/output")
    folder = "server/output/"
    surrogates = "server/surrogates/"

    for f in fold:
        try:
            print("Converting chrome-based-surrogates: website: ", f)
            # {request_id: request_url}
            request_id = request_response_dic(folder + f + "/request.json")

            files = os.listdir(folder + f + "/surrogate")
            for fil in files:
                req_id = fil.split("_")[0]
                req_url = request_id[req_id]
                create_directory_from_url(req_url, surrogates, folder + f + "/surrogate/" + fil)
        except Exception as e:
            print("Crashed chrome-based-surrogates: website: ", f, e)


if __name__ == "__main__":
    main()