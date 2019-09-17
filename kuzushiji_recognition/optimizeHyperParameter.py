import os
import random
import numpy as np

def testHyperParameter(func, filenameModel, resultsFile, testParam, testedParam, historicLoss, historicParam, parameterName, boundValue):
    change = False
    if (all([boundValue[i][0] <= testParam[i] <= boundValue[i][1] for i in range(len(testParam))])) and not testedParam[tuple(testParam)]:
        try:
            print("Test param set", testParam)
            testLoss = func('temp.h5', *testParam)
            testedParam[tuple(testParam)] = True

            if testLoss < historicLoss[-1]:
                change=True
                historicLoss.append(testLoss)
                historicParam.append(testParam)
                print("New param set", historicParam[-1])
                os.system("cp temp.h5 "+filenameModel)

        except ResourceExhaustedError:
            testedParam[tuple(testParam)] = True
            
        np.savez_compressed(resultsFile,
                            testedParam=testedParam,
                            historicParam=np.asarray(historicParam),
                            historicLoss=np.asarray(historicLoss),
                            parameterName=np.asarray(parameterName),
                            boundValue=np.asarray(boundValue))
        
    return change, testedParam, historicParam, historicLoss



def optimizeHyperParameter(func, filenameModel, resultsFile, initialValue = None, boundValue=None, parameterName=None, step=1, maxIter=30, testRandom=True):

    #Load data if exist
    startPointTested = False
    if(os.path.isfile(resultsFile)):
        raw = np.load(resultsFile)
        testedParam = raw['testedParam']
        historicParam = raw['historicParam'].tolist()
        historicLoss = raw['historicLoss'].tolist()
        
        if initialValue is None:
            startPointTested=True
            boundValue = raw['boundValue'].tolist()
            parameterName = raw['parameterName']
    else :
        historicParam = []
        historicLoss = []
    
    #Initialization if needed (no previous data or new start point)
    if not startPointTested:
        historicParam.append(initialValue)
        historicLoss.append(func(filenameModel, *historicParam[-1]))
        testedParam = np.zeros(tuple([(boundValue[i][1]+1) for i in range(len(initialValue))]), dtype=np.bool)
        testedParam[tuple(initialValue)] = True
        np.savez_compressed(resultsFile,
                            testedParam=testedParam,
                            historicParam=np.asarray(historicParam),
                            historicLoss=np.asarray(historicLoss),
                            parameterName=np.asarray(parameterName),
                            boundValue=np.asarray(boundValue))
        
    if type(step) is int:
        step = [step]*len(historicParam[-1])

    i=0
    stop=False
    while (i<maxIter) and not stop:
        i+=1
        stop=True
        
        #Iter param and test values
        for j in range(len(historicParam[-1])):
            testParam = historicParam[-1]
            
            #Test low
            testParam[j] -= step[j]
            change, testedParam, historicParam, historicLoss = testHyperParameter(func, filenameModel, resultsFile, testParam, testedParam, historicLoss, historicParam, parameterName, boundValue)
            stop = (stop and not change)
            
            #Test up
            testParam[j] += 2*step[j]
            change, testedParam, historicParam, historicLoss = testHyperParameter(func, filenameModel, resultsFile, testParam, testedParam, historicLoss, historicParam, parameterName, boundValue)
            stop = (stop and not change)
            
        #Generate a random paramset and test it
        if testRandom:
            testParam = historicParam[-1]
            for j in range(len(testParam)):
                testParam[j] = random.randint(boundValue[j][0], boundValue[j][1])
            
            change, testedParam, historicParam, historicLoss = testHyperParameter(func, filenameModel, resultsFile, testParam, testedParam, historicLoss, historicParam, parameterName, boundValue)
            stop = (stop and not change)
            

    print('\n\n\nFinal param set :')
    for j in range(len(historicParam[-1])):
        print(parameterName[j], ':', historicParam[-1][j])

    print("\nNb iteration", i, "/", maxIter)
    print("Nb case tested", np.sum(testedParam), "/", np.sum(np.ones(testedParam.shape, dtype=np.bool)))
    
    os.system("rm temp.h5")

    