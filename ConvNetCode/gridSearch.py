# Script to preform grid search over a large set of parameters, so we can compare results
# and determine the best set of hyperparameters for this problem

import tensorflow as tf

import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as callbacks

import matplotlib.pyplot as plt

from utilities import prepareTrainingData, reshapeDs, findUnusedFileName, createModelOfType 

import numpy as np
import json

## Parameters and script setup ##

# Directory that the data subfolders are stored in
dataDir = '../dataExtraction/dataDir/HandCheckedExtractedImages/'

imgExtnsion = '.jpg'

# The list of zooplankton taxa we will be training to classify among
taxaList = ['Amphipods', 'Copepods', 'FishLarvae', 'Ostracods', 'Quetognaths']

# Whether we should save the model that performs best on the validation data
saveBestModel = True
modelSaveDir = './cnnModels/'

# Hyperparameters
rngSeed = 171

numEpochs = 200

valPortion = 0.1
cnnToUse = 'mobileNet'



# Hyperparameters to search over
lrs = [ 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 4e-3, 4e-3]

batch_sizes = [1, 2, 4, 5, 10]

imageImportSizes = [224, 300, 500, 800]

simParamsDict = {'cnnToUse': cnnToUse, 
        'valPortion': valPortion, 
        'numEpochs': numEpochs,
        'rngSeed': rngSeed,
        }

# Number of output classes
numOutputClasses = len(taxaList)

# Images should be loaded with the correct color mode
# depending on the model
if(cnnToUse in ['cnn', 'vanilla_cnn', 'plain_cnn', 'cnn_withDataAug']):
    colorModeForImages = 'grayscale'
else:
    colorModeForImages = 'rgb'

simParamsDict['colorMode'] = colorModeForImages
simParamsDict['cnnToUse'] = cnnToUse

# Number of output classes
numOutputClasses = len(taxaList)


# Setting up the "grid" of grid-search fame

# Parameters in the grid are: lr, batch_size, imageImportSize
gridSearchHyperParams = []

for targLr in lrs:
    for targBatchSize in batch_sizes:
        for targImgImportSize in imageImportSizes:

            gridSearchHyperParams.append((targLr, targBatchSize, targImgImportSize))


# Declaring callbacks to use for the model training
earlyStop = callbacks.EarlyStopping(
            monitor='val_accuracy', 
            restore_best_weights=True,
            patience=15,
            verbose=1
        )

reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)

currentBestValAcc = 0

for targHyperparams in gridSearchHyperParams:
    curModel = createModelOfType(cnnToUse, targHyperparams[2], numOutputClasses, targHyperparams[0], rngSeed)

    

    train_ds = utils.image_dataset_from_directory(
        dataDir, 
        validation_split=valPortion,
        subset='training',
        seed=rngSeed,
        image_size=(targHyperparams[2], targHyperparams[2]),
        batch_size=targHyperparams[1], 
        color_mode=colorModeForImages
    )

    val_ds = utils.image_dataset_from_directory(
        dataDir,
        validation_split=valPortion,
        subset="validation",
        seed=rngSeed,
        image_size=(targHyperparams[2], targHyperparams[2]),
        batch_size=targHyperparams[1],
        color_mode=colorModeForImages
    )

    # Doing data augmentation during the model preprocessing
    train_ds_augmented = prepareTrainingData(train_ds, targHyperparams[2], shuffle=True, augment=True)

    print('Starting training of new model with hyperparams')
    print(targHyperparams)
    print()

    curHistory = curModel.fit(
        train_ds_augmented, 
        validation_data=val_ds, 
        epochs=numEpochs,
        verbose=2,
        callbacks=[earlyStop, reduce_lr_on_plateau]
    )
    
    targValAcc = np.max(curHistory.history['val_accuracy'])
    
    if(targValAcc > currentBestValAcc):
        print('Updating the best model with one that got')
        print(str(targValAcc) + ' for its validation accuracy')
        curBestModel = curModel
        curBestHist = curHistory
        currentBestValAcc = targValAcc
        simParamsDict['bestModelLr'] = targHyperparams[0]


print('Best Validation accuracy was: ' + str(currentBestValAcc))


if(saveBestModel == True):
    
    newModelWeights_filename = findUnusedFileName(modelSaveDir, 'cnnGridSearchModel_', '.h5')
    
    curBestModel.save_weights(modelSaveDir + newModelWeights_filename)
    
    beforeFileType = newModelWeights_filename[:newModelWeights_filename.rfind('.')]
    
    modelParamsFilename = modelSaveDir + beforeFileType + '.txt'

    with open(modelParamsFilename, 'w') as fp:
        json.dump(simParamsDict, fp, indent=4)


    print('saving model to: ' + newModelWeights_filename)
    print('saving model description to: ' + modelParamsFilename)


