## Script to check how well the model performed on data from the training set under
## specific augmentation processes
from CNNModel import initializeCNN_withAug, initializeCNN, initialize_MobileNet, initialize_ConvNeXt, initialize_MobileNetV3, initialize_InceptionNet
from utilities import createModelOfType, prepareTrainingData 
import tensorflow.keras.utils as utils
import tensorflow.keras as keras
import numpy as np
import json

# Parameters of the simulation
dataDir = '../dataExtraction/dataDir/HandCheckedExtractedImages/'
modelDir = './cnnModels/'
modelToLoad = 'cnnModel_45'


# Modifications to apply to the training data
shuffleVals = [True, False]
augmentVals = [True, False]

dataMods = []

for targShuffle in shuffleVals:
    for targAugment in augmentVals:
        dataMods.append((targShuffle, targAugment))



with open(modelDir + modelToLoad + '.txt', 'r') as fp:
    simDict = json.load(fp)

lr = simDict['lr']
rngSeed = simDict['rngSeed']
imageImportSize = simDict['imageImportSize']
batchSize = simDict['batchSize']
valPortion = simDict['valPortion']
cnnToUse = simDict['cnnToUse']
colorMode = simDict['colorMode']


train_ds = utils.image_dataset_from_directory(
    dataDir,
    validation_split=valPortion,
    subset="training",
    seed=rngSeed,
    image_size=(imageImportSize, imageImportSize),
    batch_size=batchSize,
    color_mode=colorMode
)


# Setting up the CNN to be one of the prespecified models
evalCNN = createModelOfType(cnnToUse, imageImportSize, numOutputClasses, lr, rngSeed)


evalCNN.load_weights(modelDir + modelToLoad + '.h5')

for targMod in dataMods:
    trainDs = prepareTrainingData(train_ds, shuffle=targMod[0], augment=targMod[1], concatenate=False)

    trainLoss, trainAcc = evalCNN.evaluate(trainDs)
    
    print('Train accuracy was: ' + str(trainAcc))
    print('When the data mods were ')
    print(targMod)



