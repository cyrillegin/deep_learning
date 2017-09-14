# Jacob Bakarich
# Cyrille Gindreau
# EE 490
# 9/14/2017
# Project 1

import numpy as np
import matplotlib.pyplot as plt
import math
import json


def setup():
    inputs = np.empty([50, 2])
    z = np.empty([50, 1])  # logit for use in calculations
    calculatedOutput = np.empty([50, 1])  # used for storing final output
    finalOutput = np.empty([50, 1])  # output using activation function
    deltaWeights = np.empty([50, 2])

    inputs[:, 0], inputs[:, 1], testOutputs = np.loadtxt("fruit_data.txt", unpack=True)  # load inputs
    weights = np.array(([1, 1]))
    # no bias
    # epsilon is 1
    obj = {}
    obj["weights"] = weights
    obj["inputs"] = inputs
    obj["z"] = z
    obj["calculatedOutput"] = calculatedOutput
    obj["deltaWeights"] = deltaWeights
    obj["finalOutput"] = finalOutput
    obj["testOutputs"] = testOutputs
    return obj


def takeStep(StepInfo):
    error = 0

    for i in range(0, 50):
        # first calculate z value
        StepInfo["z"][i] = (StepInfo["weights"][0]*StepInfo["inputs"][i, 0]) + (StepInfo["weights"][1]*StepInfo["inputs"][i, 1])

        # calculate activation function value
        StepInfo["calculatedOutput"][i] = 1/(1 + math.exp(-StepInfo["z"][i]))  # Final output through sigmoid

        # calculate error
        error += (0.5)*((StepInfo["testOutputs"][i] - StepInfo["calculatedOutput"][i])**2)

        # now ready to calculate weight difference for each weight value
        for j in range(0, 2):
            StepInfo["deltaWeights"][i, j] = StepInfo["inputs"][i, j] * StepInfo["calculatedOutput"][i] * ((1 - StepInfo["calculatedOutput"][i])*(StepInfo["testOutputs"][i] - StepInfo["calculatedOutput"][i]))

    StepInfo["deltaWeightsSum"] = StepInfo["deltaWeights"].sum(axis=0)
    StepInfo["weights"] = StepInfo["deltaWeightsSum"] + StepInfo["weights"]

    print("Calculated Weights:", StepInfo["weights"], "Error:", (error*0.5))
    return StepInfo


def getZ(arr):
    error = 0
    final = []
    for i in range(0, 25):
        # first calculate z value
        arr["z"][i] = (arr["weights"][0]*arr["inputs"][i][0]) + (arr["weights"][1]*arr["inputs"][i][1])

        # calculate activation function value
        arr["calculatedOutput"][i] = 1/(1 + math.exp(-arr["z"][i]))  # Final output through sigmoid

        # calculate error
        error += (0.5)*((arr["testOutputs"][i] - arr["calculatedOutput"][i])**2)

        # now ready to calculate weight difference for each weight value
        for j in range(0, 2):
            arr["deltaWeights"][i, j] = arr["inputs"][i][j] * arr["calculatedOutput"][i] * ((1 - arr["calculatedOutput"][i])*(arr["testOutputs"][i] - arr["calculatedOutput"][i]))

        arr["deltaWeightsSum"] = arr["deltaWeights"].sum(axis=0)
        arr["weights"] = arr["deltaWeightsSum"] + arr["weights"]

        final.append([arr['inputs'][i][0], arr['inputs'][i][1], float(error*0.5)])
    return final
