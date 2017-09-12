#Jacob Bakarich
#Cyrille Gindreau
#EE 490
#9/14/2017
#Project 1

import numpy as np
import matplotlib.pyplot as plt
import math

inputs = np.empty([50, 2])
z = np.empty([50, 1]) #logit for use in calculations
calculatedOutput = np.empty([50, 1]) #used for storing final output
finalOutput = np.empty([50, 1]) #output using activation function
deltaWeights = np.empty([50, 2])



inputs[:,0], inputs[:,1], testOutputs = np.loadtxt("fruit_data.txt", unpack = True) #load inputs
weights = np.array(([1, 1]))
#no bias
#epsilon is 1

for iterations in range(0, 20000): #call for as many iterations as desired
    error = 0; #reset error amount for every new iteration

    for i in range(0, 50):
        #first calculate z value
        z[i] = (weights[0]*inputs[i, 0]) + (weights[1]*inputs[i, 1])

        #calculate activation function value
        calculatedOutput[i] = 1/(1 + math.exp(-z[i])) #Final output through sigmoid


        #calculate error
        error += (0.5)*((testOutputs[i] - calculatedOutput[i])**2)


        #now ready to calculate weight difference for each weight value
        for j in range (0, 2):
            deltaWeights[i, j] = inputs[i, j] * calculatedOutput[i] * ((1 - calculatedOutput[i])*(testOutputs[i] - calculatedOutput[i]))

    deltaWeightsSum = deltaWeights.sum(axis = 0);
    weights = deltaWeightsSum + weights

    print("Calculated Weights:", weights, "Error:", (error*0.5))
