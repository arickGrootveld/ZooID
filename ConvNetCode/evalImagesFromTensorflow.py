# This script exists just to check that the images as they are put into the model
# through tensorflow make sense


import tensorflow.keras.utils as utils
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Params of the sim
imgLoadDir = '../dataExtraction/dataDir/extractedImages/'
imgSaveDir = './figures/'
rngSeed = 24
imageImportSize = 224
colorModeForImages = 'rgb'

numImagesToSave = 10

# Using this to specify that labels should be:
#   Amphipods: 0
#   Copepods: 1
#   FishLarvae: 2
#   Ostracods: 3
#   Quetognaths: 4
taxaList = ['Amphipods', 'Copepods', 'FishLarvae', 'Ostracods', 'Quetognaths']


# Loading the dataset
dataSet = utils.image_dataset_from_directory(
    imgLoadDir,
    validation_split=0.1,
    subset="validation",
    seed=rngSeed,
    image_size=(imageImportSize, imageImportSize),
    batch_size=1,
    color_mode=colorModeForImages,
    class_names=taxaList
)

dataSetFilepaths = dataSet.file_paths

dsIter = iter(dataSet)

for i in range(numImagesToSave):
    targImgArray = next(dsIter)
    targFilepath = dataSetFilepaths[i]

    targImg = targImgArray[0]

    targImg_Formatted = (targImg.numpy()/ 255)[0,:,:,:]

    plt.imshow(targImg_Formatted)
    plt.title('Image ' + str(i) + ': ' + taxaList[targImgArray[1].numpy()[0]])

    plt.savefig(imgSaveDir + 'tensorflowImage_' + str(i) + '.jpg')
    
    plt.clf()

    extractedImage = np.asarray(Image.open(targFilepath))
    plt.imshow(extractedImage)
    plt.title(targFilepath)
    plt.savefig(imgSaveDir + 'extractedImage_' + str(i) + '.jpg')
    plt.clf()





