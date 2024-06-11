"""
This files contain the logic to gather tracking and mixed scripts or methods for blocking experiments.
"""
import math
import numpy as np
import pandas as pd
import json
import os


def getInitiatorScript(stack):
    if len(stack["callFrames"]) != 0:
        return stack["callFrames"][0]["url"]
    else:
        return getInitiatorScript(stack["parent"])


def getInitiatorMethod(stack):
    if len(stack["callFrames"]) != 0:
        return (
            stack["callFrames"][0]["url"]
            + "@"
            + stack["callFrames"][0]["functionName"]
            + "@"
            + str(stack["callFrames"][0]["lineNumber"])
            + "@"
            + str(stack["callFrames"][0]["columnNumber"])
        )
    else:
        return getInitiatorMethod(stack["parent"])


def addScript(script, key, tc, fc, topURL):
    try:
        if key not in script.keys():
            script[key] = [0, 0, 0, []]  # tc, fc, log10(tc/fc), toplevelurl
        script[key][0] += tc
        script[key][1] += fc
        script[key][2] = math.log(
            (script[key][0] + 0.0000001) / (script[key][1] + 0.0000001), 10
        )
        if topURL not in script[key][3]:
            script[key][3].append(topURL)
    except:
        pass


def addMethod(method, key, tc, fc, topURL):
    try:
        if key not in method.keys():
            method[key] = [0, 0, 0, [], []]  # tc, fc, log10(tc/fc), toplevelurl
        method[key][0] += tc
        method[key][1] += fc
        method[key][2] = math.log(
            (method[key][0] + 0.0000001) / (method[key][1] + 0.0000001), 10
        )
        if topURL not in method[key][3]:
            method[key][3].append(topURL)
    except:
        pass


def getScriptsMethods():
    script = {}
    method = {}

    path = "/Users/haadi/Desktop/webpage-crawler-extension/server/output"

    fold = os.listdir(path)
    for f in fold:
        if ".com" in f:
            if os.path.isfile(path + "/" + f + "/label_request.json"):
                print(f)
                df = pd.read_json(path + "/" + f + "/label_request.json")
                for index, dataset in df.iterrows():
                    try:
                        if dataset["call_stack"]["type"] == "script":
                            if (
                                dataset["easylistflag"] == 1
                                or dataset["easyprivacylistflag"] == 1
                                or dataset["ancestorflag"] == 1
                            ):
                                addScript(
                                    script,
                                    getInitiatorScript(dataset["call_stack"]["stack"]),
                                    1,
                                    0,
                                    dataset["top_level_url"],
                                )
                                addMethod(
                                    method,
                                    getInitiatorMethod(dataset["call_stack"]["stack"]),
                                    1,
                                    0,
                                    dataset["top_level_url"],
                                )
                            else:
                                addScript(
                                    script,
                                    getInitiatorScript(dataset["call_stack"]["stack"]),
                                    0,
                                    1,
                                    dataset["top_level_url"],
                                )
                                addMethod(
                                    method,
                                    getInitiatorMethod(dataset["call_stack"]["stack"]),
                                    0,
                                    1,
                                    dataset["top_level_url"],
                                )
                    except:
                        pass
    trackingScripts = []
    mixedScripts = []
    mixedScriptMethod = {}
    for s in script.keys():
        if script[s][2] >= -2 and script[s][2] <= 2:
            if s != "" and "http" in s:
                if s.find("http://") == 0 or s.find("https://") == 0:
                    mixedScripts.append(s)
        elif script[s][2] > 2:
            if s != "" and "http" in s:
                if s.find("http://") == 0 or s.find("https://") == 0:
                    trackingScripts.append(s)
    for m in method:
        lst = m.split("@")
        if lst[0] != "" and "http" in lst[0] and lst[0] in mixedScripts:
            if lst[0].find("http://") == 0 or lst[0].find("https://") == 0:
                if method[m][2] > 2:
                    if lst[0] not in mixedScriptMethod.keys():
                        mixedScriptMethod[lst[0]] = []
                    mixedScriptMethod[lst[0]].append([lst[1], lst[2], lst[3]])
                elif method[m][2] >= -2 and method[m][2] <= 2:
                    if lst[0] not in mixedScriptMethod.keys():
                        mixedScriptMethod[lst[0]] = []
                    mixedScriptMethod[lst[0]].append([lst[1], lst[2], lst[3]])

    # json.dump(script, open(path+'/'+ f + "/blockScripts.json", "w"))
    # json.dump(method, open(path+'/'+ f + "/blockMethods.json", "w"))

    with open(path + "/lst/trackingScripts.txt", "w") as log:
        log.write(str(trackingScripts))
        log.close()
    with open(path + "/lst/mixedScripts.txt", "w") as log:
        log.write(str(mixedScripts + trackingScripts))
        log.close()
    with open(path + "/lst/mixedScriptMethods.txt", "w") as log:
        log.write(str(mixedScriptMethod))
        log.close()


getScriptsMethods()
