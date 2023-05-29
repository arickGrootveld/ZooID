import tensorflow as tf
import keras.layers as layers
import os

from CNNModel import initializeCNN_withAug, initializeCNN, initialize_MobileNet, initialize_ConvNeXt, initialize_MobileNetV3, initialize_InceptionNet, initialize_MobileNet_2Deep
AUTOTUNE = tf.data.AUTOTUNE

# A function to prepare the data for training
def prepareTrainingData(ds, inputSize, shuffle=False, augment=False, concatenate=False):

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(inputSize, inputSize),
        layers.Rescaling(1./255)
    ])

    ds = ds.map(lambda x, y : (resize_and_rescale(x), y), 
                num_parallel_calls=AUTOTUNE)
    
    if(shuffle==True):
        ds.shuffle(1000)
    

    if(augment == True):
        if(concatenate == True):
            nonAugmentedData = ds
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])
        ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)
        
        if(concatenate == True):
            ds = ds.concatenate(nonAugmentedData)

    
    

    return(ds.prefetch(buffer_size=AUTOTUNE))



def reshapeDs(ds, inputSize):
    # Adding an extra dimension for the convolutional layers to work
    reshaper = tf.keras.Sequential([
        layers.Reshape((inputSize, inputSize, 1), input_shape=(inputSize, inputSize))
    ])
    
    ds = reshaper(ds)

    return(ds)




def findUnusedFileName(fileSaveDir, fileNameTemplate, extension):
    existingFileNames = os.listdir(fileSaveDir)

    newNameFound = False
    iterator = -1

    while(newNameFound == False):
        iterator = iterator + 1

        newName = fileNameTemplate + str(iterator) + extension

        if(newName not in existingFileNames):
            newNameFound = True
            finalFileName = newName
        
    return(finalFileName)



def createModelOfType(cnnToUse, imageImportSize, numOutputClasses, lr, rngSeed): 
    
    # Setting up the CNN to be one of the prespecified models
    if(cnnToUse in ['cnn', 'vanilla_cnn', 'plain_cnn']):
        taxaCNN = initializeCNN((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr, seed=rngSeed)
    elif(cnnToUse in ['cnn_withDataAug']):
        taxaCNN = initializeCNN_withAug((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
    elif(cnnToUse in ['mobileNet', 'mobileNetV2']):
        taxaCNN = initialize_MobileNet((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
    elif(cnnToUse in ['mobileNetV3']):
        taxaCNN = initialize_MobileNetV3((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
    
    elif(cnnToUse in ['ConvNeXt']):
        taxaCNN = initialize_ConvNeXt((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)

    elif(cnnToUse in ['inceptionNet']):
        taxaCNN = initialize_InceptionNet((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)
    
    elif(cnnToUse in ['mobileNet2Deep', 'mobileNet_2Deep']):
        taxaCNN = initialize_MobileNet_2Deep((imageImportSize, imageImportSize), num_classes=numOutputClasses, lr=lr)

    else:
        print('The \"cnnToUse\" variable is: ' + cnnToUse)
        raise Exception('Not a valid model name')
    
        

    return(taxaCNN)


