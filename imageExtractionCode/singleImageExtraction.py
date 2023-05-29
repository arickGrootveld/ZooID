# Script to extract a single image to see what it will look like

import numpy as np
import os
import cv2
import glob
from PIL import Image


# Importing the helper function from the other script so we can have it here to use
from formattingAndExtraction import individualImageProcessingRouting


# Parameters of the script
targetFile = './dataDir/originalImages/Amphipods/Amfipodo 19.jpg'

saveLoc = './dataDir/individualFileTests/'



extractedImages = individualImageProcessingRouting(targetFile)

for targExtImageInd, targExtImage in enumerate(extractedImages):
    saveFilename = saveLoc + 'extractedImage' + str(targExtImageInd) + '.jpg'
    
    print('Saving extracted image to: ' + saveFilename)
    cv2.imwrite(saveFilename, targExtImage)
