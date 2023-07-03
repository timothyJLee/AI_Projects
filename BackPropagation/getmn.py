import numpy as np

def getmnist():
    import random, pickle
    with open('./littlemnist.pkl', 'rb') as f:
        (trainX, trainY), (validX, validY), (testX, testY) = pickle.load(f,encoding= 'latin1')
    print ('loaded data')
    train = [(x,y)for x,y in zip (trainX, trainY)]
    valid = [(x,y)for x,y in zip (validX, validY)]
    test =  [(x,y)for x,y in zip (testX, testY)]
    return train, valid, test

