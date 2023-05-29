# Script to implement a simplistic UI for the model
from modelAndData_Utils import setupModel, loadImage, modelPredOnImage,\
                                individualImageExtraction, \
                                convertExtractedImageToTensor
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

# Parameter of the script
imageExtractionAndInference = True

# Path to the .hdf5 file that the model weights are stored in
modelWeightsFilepath = './model_files/cnnModel_48.h5'

if(__name__ == '__main__'):
    print('Basic implementation of a model UI')

    if(imageExtractionAndInference == False):
        inferenceModel = setupModel(modelWeightsFilepath)

        root = tk.Tk()
        root.withdraw()

        targFilepath = filedialog.askopenfilename()


        targImage = loadImage(targFilepath)

        predTaxa = modelPredOnImage(inferenceModel, targImage)

        print()
        print()
        print('The predicted taxa is: ' + predTaxa)
        print()
        print()
    else:
        # If we get here, then we want to prompt the 
        # user for the image and then extract it as well
        inferenceModel = setupModel(modelWeightsFilepath)

        root = tk.Tk()
        root.withdraw()

        targFilepath = filedialog.askopenfilename()

        extractedImages = individualImageExtraction(targFilepath)

        extractedImageLabels = []

        for targExtractedImage in extractedImages:
            convertedImage = convertExtractedImageToTensor(targExtractedImage)

            predTaxa = modelPredOnImage(inferenceModel, convertedImage)

            print('predicted taxa is: ' + predTaxa)

            # Showing the extracted image before performing inference
            plt.imshow(targExtractedImage)
            plt.title('Inferred Image Label: ' + predTaxa)
            plt.show()

            plt.clf()


