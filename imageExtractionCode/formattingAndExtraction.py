# Script to convert the original images into segmented sub-images
# that we can use for training

import numpy as np
import os
import cv2
import glob
from PIL import Image

## Parameters of the script
originalImages_Dir = './dataDir/originalImages/'
extractedImages_Dir = './dataDir/extractedImages/'

dataClassLabels = ['Amphipods', 'Copepods', 'FishLarvae', 'Ostracods', 'Quetognaths']




## Checking if the correct directories exist in the data extraction folder
for labName in dataClassLabels:
    # Checking if the directory for this class label exists
    dirExists = os.path.exists(extractedImages_Dir + labName)
    if(dirExists == False):
        os.mkdir(extractedImages_Dir + labName)

## Helper function for the image processing
def individualImageProcessingRouting(targImagePath):

    ################# KMeans Clustering and then Canny Edge Detection #################
    sample_image = cv2.imread(targImagePath)


    img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10

    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))


    ################# Applying Gaussian Blur to Canny Edges #################

    # Canny Edge Detection
    canny_edges_afterKMeans = cv2.Canny(image=result_image, threshold1=80, threshold2=110) # Canny Edge Detection

    # Bluring the Canny edges to make it easier to get the convex hull
    dst = cv2.GaussianBlur(canny_edges_afterKMeans,(3,3),1)

    ################# Extracting contours #################
    contours,_ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i]))

    ################# Image Masking #################

    # Array to hold the images after the masking is complete
    maskedImages = []

    copyOfSampleImage = sample_image.copy()
    # Grayscale copy to extract parts from
    grayScaleCopy = cv2.cvtColor(copyOfSampleImage, cv2.COLOR_BGR2GRAY)

    for hullInd in range(len(hull)):
        # Setting an arbitrary threshold to only get the larger portions of the image
        if(len(hull[hullInd]) > 25):
            # Reseting the image each time so we don't accidentally concatenate the images
            newBlankImage = np.zeros(sample_image.shape, dtype=sample_image.dtype)

            targMask = cv2.cvtColor(cv2.drawContours(newBlankImage, hull, hullInd, (255,255,255), -1, 8), cv2.COLOR_BGR2GRAY)

            maskedImage = cv2.bitwise_and(grayScaleCopy, grayScaleCopy, mask=targMask)

            maskedImages.append(maskedImage)


    ################# Dealing with Overlaps in Masks ################# 

    # Removing redundant convex hulls (Expecting each of the regions of interest to be disjoint)
    ### Very very inefficient way of doing this, but since this only needs to be run once it should
    ### be fine
    arrayOfIndexGroups = []
    for firstInd in range(len(maskedImages)):
        initialImage = maskedImages[firstInd]
        firstIndsInts = [firstInd]
        # Looping over each element, and determining which elements aren't disjoint, and what indexes
        # they are not disjoint with
        for secondInd in range(firstInd+1, len(maskedImages)):
            # Now we are looping over every element, and then comparing it with every element after it 
            # in the array
            secondaryImage = maskedImages[secondInd]
            targIntersections = cv2.bitwise_and(initialImage, secondaryImage)
            # This if statement checks if they have a non-zero intersection
            if(np.count_nonzero(targIntersections) != 0):
            # If we get below here, then we have a found a place where two or more masks intersect
                notYetIn = True
                # Going through the "arrayOfIndexGroups" to find out if 
                # either index is already associated to a group
                for groupIndex in range(len(arrayOfIndexGroups)):
                    targGroup = arrayOfIndexGroups[groupIndex]

                    if(firstInd in targGroup):
                        # If the first group already exists in here, 
                        # then we just see if we need to append the secondary index or not
                        notYetIn = False
                        if(not secondInd in targGroup):
                            # Updating the group to include the 2nd index if its not already in here
                            targGroup.append(secondInd)
                            arrayOfIndexGroups[groupIndex] = targGroup

                    if(notYetIn ==  True):
                        # If the first index isn't already in a group, then we create a new group with
                        # both of them
                        newGroup = [firstInd, secondInd]
                        arrayOfIndexGroups.append(newGroup)

    # Creating a flattened version of the index groups so we can easily find the non-overlapping ones
    flattenedIndexGroups = [index for sublist in arrayOfIndexGroups for index in sublist]
    # Getting array of indexes of the masks with no intersection
    goodArrayImageInds = [maskInd for maskInd in np.arange(len(maskedImages)) if(maskInd not in flattenedIndexGroups)]
    # Creating the new array with only disjoint masks
    cleanImages = [maskImage for ind, maskImage in enumerate(maskedImages) if(ind in goodArrayImageInds)]
    # Adding to the "cleanImages" array with the bitwise or of each group
    for targGroup in arrayOfIndexGroups:
        holderImage = np.zeros(grayScaleCopy.shape, dtype=grayScaleCopy.dtype)
        for targInd in targGroup:
            targImage = maskedImages[targInd]
            holderImage = np.bitwise_or(holderImage, targImage)
        cleanImages.append(holderImage)


    ################# Cropping Images to Minimum Rectangle ################# 

    croppedImages = []


    for targImage in cleanImages:
        nonZInds = np.nonzero(targImage)
        minX = np.min(nonZInds[0])
        maxX = np.max(nonZInds[0])

        minY = np.min(nonZInds[1])
        maxY = np.max(nonZInds[1])

        croppedImage = targImage[minX:maxX, minY:maxY]

        croppedImages.append(croppedImage)

    return(croppedImages)


if(__name__ == '__main__'):

    ## Converting all .tif files to jpgs to make life easier
    inputFileExtension = '.tif'
    outputFileExtension = '.jpg'

    targInputFiles = glob.glob(originalImages_Dir + '*/*' + inputFileExtension)

    for targInputFilepath in targInputFiles:

        # For each .tif file, save an alternate .jpg file in its place
        targExtInd = targInputFilepath.rfind(inputFileExtension)
        targOutFilepath = targInputFilepath[:targExtInd] + outputFileExtension

        if(not os.path.isfile(targOutFilepath)):
            targImage = Image.open(targInputFilepath)

            print('Saving new image to: ' + targOutFilepath)
            targImage.save(targOutFilepath, "JPEG", quality=100)

        else:
            print('JPG already exists at: ' + targOutFilepath)





    ## Loading the data
    imageFileFormat = '.jpg'

    targImageFilenames = glob.glob(originalImages_Dir + '*/*' + imageFileFormat)


    labelImagesDict = {}
    for targLabel in dataClassLabels:
        targLabelArray = [targFn for targFn in targImageFilenames if(targLabel in targFn[ targFn.rfind('/', 0, targFn.rfind('/')) + 1: targFn.rfind('/') ] )]
        labelImagesDict[targLabel] = targLabelArray

    print('Starting the image processing stuff')

## Processing the images
    for targLabelName in dataClassLabels:
        targLabelArray = labelImagesDict[targLabelName]
        for targFilepath in targLabelArray:
            targFilename = targFilepath[targFilepath.rfind('/')+1:targFilepath.rfind('.')]
            saveFilenameTemplate = extractedImages_Dir + targLabelName + '/' + targFilename + '_extractedImage'
            extractedImages = individualImageProcessingRouting(targFilepath)

            for extractInd, extractedImage in enumerate(extractedImages):
                saveFilename = saveFilenameTemplate + str(extractInd) + imageFileFormat

                print(saveFilename)

                cv2.imwrite(saveFilename, extractedImage)

