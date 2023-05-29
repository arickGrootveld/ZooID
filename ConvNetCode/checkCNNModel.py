## Script to check the outputs of one of the CNN models
from CNNModel import initializeCNN_withAug, initializeCNN, initialize_MobileNet, initialize_ConvNeXt, initialize_MobileNetV3, initialize_InceptionNet
from utilities import createModelOfType, prepareTrainingData 
import tensorflow.keras.utils as utils
import tensorflow.keras as keras
import numpy as np
import json

# Parameters of the simulation
dataDir = '../dataExtraction/dataDir/HandCheckedExtractedImages/'
modelDir = './cnnModels/'
modelToLoad = 'cnnModel_44'

saveResults = False     # Variable to say whether we should save the predictions and results

printConfMat = True     # Variable to say whether we should print the confusion matrix

evalResults = True      # Whether to show the accuracy of the model on the dataset

augmentData = True     # Whether to augment the data while checking the results


with open(modelDir + modelToLoad + '.txt', 'r') as fp:
    simDict = json.load(fp)

imagesToGoThrough = 1
numOutputClasses = 5

lr = simDict['lr']
rngSeed = simDict['rngSeed']
imageImportSize = simDict['imageImportSize']
batchSize = simDict['batchSize']
# batchSize = 1
valPortion = simDict['valPortion']
# valPortion = 0.99
cnnToUse = simDict['cnnToUse']
colorMode = simDict['colorMode']


val_ds = utils.image_dataset_from_directory(
    dataDir,
    validation_split=valPortion,
    subset="training",
    seed=rngSeed,
    image_size=(imageImportSize, imageImportSize),
    batch_size=batchSize,
    color_mode=colorMode
)

val_ds_aug = prepareTrainingData(val_ds, imageImportSize, shuffle=True, augment=augmentData, concatenate=False)

# Setting up the CNN to be one of the prespecified models
evalCNN = createModelOfType(cnnToUse, imageImportSize, numOutputClasses, lr, rngSeed)


evalCNN.load_weights(modelDir + modelToLoad + '.h5')


ds_iter = iter(val_ds)

for i in range(imagesToGoThrough):
    targImageLabelArray = next(ds_iter)

    targOutput = evalCNN.predict(targImageLabelArray[0])
    
    print('This is image: ' + str(i))
    print('The output of the model for this image is: ')
    print(targOutput)

    print('The correct label for this image is: ' + str(keras.backend.get_value(targImageLabelArray[1])))
    print()


modelPredictions = evalCNN.predict(val_ds)

# print(modelPredictions)

# Saving the labels and the predictions to files
if(saveResults == True):
    np.savetxt("./predictions.txt", modelPredictions)
    np.savetxt("./dataLabels.txt", labelArray)

if(evalResults == True):
    evalLoss, evalAcc = evalCNN.evaluate(val_ds)
    
    print('The accuracy of the model on this dataset was: ' + str(evalAcc))

ds_iter = iter(val_ds)

labelArray = np.zeros(len(val_ds))

for i in range(len(val_ds)):
    targImageLabelArray = next(ds_iter)

    targLabel = targImageLabelArray[1]
    labelArray[i] = targLabel.numpy()[0]



# Saving the labels and the predictions to files
if(saveResults == True):
    np.savetxt("./predictions.txt", modelPredictions)
    np.savetxt("./dataLabels.txt", labelArray)




if(printConfMat == True):
    modelPredsOneHot = np.argmax(modelPredictions, axis=1)
    from tensorflow.math import confusion_matrix

    confMat = confusion_matrix(modelPredsOneHot, labelArray)
    print('The confusion matrix is')
    print('(Note that a row corresponds to model classifications, and columns correspond to labels)')
    print(confMat)


