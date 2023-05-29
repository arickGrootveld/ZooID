#!/usr/bin/env python


## Importing libraries ##
import tensorflow as tf

# import tensorflow.keras.utils.image_dataset_from_directory
import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as callbacks

import matplotlib.pyplot as plt

from utilities import prepareTrainingData, reshapeDs, findUnusedFileName, createModelOfType
from CNNModel import initializeCNN_withAug, initializeCNN, initialize_MobileNet, initialize_ConvNeXt, initialize_MobileNetV3, initialize_InceptionNet, initialize_MobileNet_2Deep

import json

## Parameters and script setup ##

# Directory that the data subfolders are stored in
dataDir = '../dataExtraction/dataDir/HandCheckedExtractedImages/'

imgExtnsion = '.jpg'

# The list of zooplankton taxa we will be training to classify among
taxaList = ['Amphipods', 'Copepods', 'FishLarvae', 'Ostracods', 'Quetognaths']


# Parameter telling us whether to plot the models performance or not
plotPerf = False

# Parameter telling us whether we should save the model or not
saveModel = True
modelSaveDir = './cnnModels/'

# Hyperparameters
rngSeed = 502
# Batch size to use for the data
batchSize = 1
# The square dimension we want the images to be before going through the CNN
imageImportSize = 224
# Number of epochs to train for
numEpochs = 200
# Validation split size
valPortion = 0.1
# String specifying the model that was used
cnnToUse = 'cnn'
# The initial learning rate of the model
lr = 5e-6
# Parameter to specify whether to augment the data or not
augmentData = True
# Parameter to specify whether we should concatenate the original 
# dataset onto the end of the augmented data if augmentData is True
concatenateNonAugmentedData = False

# Variable to hold all the hyperparameters of the model for later storage
simParamsDict = {
        'rngSeed': rngSeed,
        'batchSize': batchSize,
        'imageImportSize': imageImportSize,
        'numEpochs': numEpochs,
        'valPortion': valPortion,
        'cnnToUse': cnnToUse,
        'lr': lr,
        'augmentData': augmentData,
        'concatenateNonAugmentedData': concatenateNonAugmentedData
        }

# Number of output classes
numOutputClasses = len(taxaList)


## Loading data ##

# Got image loading code from: https://www.tensorflow.org/tutorials/load_data/images

# Images should be loaded with the correct color mode
# depending on the model
if(cnnToUse in ['cnn', 'vanilla_cnn', 'plain_cnn', 'cnn_withDataAug']):
    colorModeForImages = 'grayscale'
else:
    colorModeForImages = 'rgb'

simParamsDict['colorMode'] = colorModeForImages


train_ds = utils.image_dataset_from_directory(
    dataDir, 
    validation_split=valPortion,
    subset='training',
    seed=rngSeed,
    image_size=(imageImportSize, imageImportSize),
    batch_size=batchSize, 
    color_mode=colorModeForImages
)

val_ds = utils.image_dataset_from_directory(
    dataDir,
    validation_split=valPortion,
    subset="validation",
    seed=rngSeed,
    image_size=(imageImportSize, imageImportSize),
    batch_size=batchSize,
    color_mode=colorModeForImages
)


# Doing data augmentation during the model preprocessing

train_ds_augmented = prepareTrainingData(train_ds, imageImportSize, shuffle=True, augment=augmentData,concatenate=concatenateNonAugmentedData)

val_ds_augmented = prepareTrainingData(val_ds, imageImportSize, shuffle=False, augment=False, concatenate=False)

# Writing callbacks to get some information about the model training
class printLrEachEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()
        currentLr = tf.keras.backend.get_value(self.model.optimizer.lr(self.model.optimizer.iterations))
        print('lr on epoch ' +  str(epoch) + ' is: ')
        print(currentLr)

reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.5, 
            patience=5,
            verbose=1
        )

# Declaring callbacks to use for the model training
earlyStop = callbacks.EarlyStopping(
            monitor='val_accuracy', 
            restore_best_weights=True,
            patience=20,
            verbose=1
        )
## Initializing the CNN and then performing training

# Setting up the CNN to be one of the prespecified models
#if(cnnToUse in ['cnn', 'vanilla_cnn', 'plain_cnn']):
#    taxaCNN = initializeCNN((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr, seed=rngSeed)
#elif(cnnToUse in ['cnn_withDataAug']):
#    taxaCNN = initializeCNN_withAug((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
#elif(cnnToUse in ['mobileNet', 'mobileNetV2']):
#    taxaCNN = initialize_MobileNet((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
#elif(cnnToUse in ['mobileNetV3']):
#    taxaCNN = initialize_MobileNetV3((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
#    
#elif(cnnToUse in ['ConvNeXt']):
#    taxaCNN = initialize_ConvNeXt((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
#
#elif(cnnToUse in ['inceptionNet']):
#    taxaCNN = initialize_InceptionNet((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
#
#else:
#    print('The \"cnnToUse\" variable is: ' + cnnToUse)
#    raise Exception('Not a valid model name')


taxaCNN = createModelOfType(cnnToUse, imageImportSize, numOutputClasses, lr, rngSeed)

history = taxaCNN.fit(
    train_ds_augmented, 
    validation_data=val_ds_augmented, 
    epochs=numEpochs,
    verbose=2,
    callbacks=[earlyStop, reduce_lr_on_plateau]
#    callbacks = [printLrEachEpoch]
#    callbacks = [reduce_lr_on_plateau]
)


## Visualizing the model over the training time
if(cnnToUse in ['inceptionNet']):
    simParamsDict['loss'] = history.history['loss']
    simParamsDict['val_loss'] = history.history['val_loss']

    simParamsDict['dense_1_loss'] = history.history['dense_1_loss']
    simParamsDict['dense_3_loss'] = history.history['dense_3_loss']
    simparamsDict['dense_4_loss'] = history.history['dense_4_loss']

    simParamsDict['dense_1_accuracy'] = history.history['dense_1_accuracy']
    simParamsDict['dense_3_accuracy'] = history.history['dense_3_accuracy']
    simParamsDict['dense_4_accuracy'] = history.history['dense_4_accuracy']

    simParamsDict['val_dense_1_accuracy'] = history.history['val_dense_1_accuracy']
    simParamsDict['val_dense_3_accuracy'] = history.history['val_dense_3_accuracy']
    simParamsDict['val_dense_4_accuracy'] = history.history['val_dense_4_accuracy']

    simParamsDict['val_dense_1_loss'] = history.history['val_dense_1_loss']
    simParamsDict['val_dense_3_loss'] = history.history['val_dense_3_loss']
    simParamsDict['val_dense_4_loss'] = history.history['val_dense_4_loss']

else:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    simParamsDict['acc'] = acc
    simParamsDict['val_acc'] = val_acc
    simParamsDict['loss'] = loss
    simParamsDict['val_loss'] = val_loss

epochs_range = range(numEpochs)

if(plotPerf == True):

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if(saveModel == True):
    newModelWeights_filename = findUnusedFileName(modelSaveDir, 'cnnModel_', '.h5')
    
    taxaCNN.save_weights(modelSaveDir + newModelWeights_filename)
    
    beforeFileType = newModelWeights_filename[:newModelWeights_filename.rfind('.')]
    
    modelParamsFilename = modelSaveDir + beforeFileType + '.txt'

    with open(modelParamsFilename, 'w') as fp:
        json.dump(simParamsDict, fp, indent=4)


    print('saving model to: ' + newModelWeights_filename)
    print('saving model description to: ' + modelParamsFilename)





