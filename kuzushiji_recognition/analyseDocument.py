import scipy


def analyseDocument(importFile, outputFile, folderImage, unicodeFile, modelCaracterEmplacement, modelCaracterRecognition,
                    caracterEmplacementRes = (1024, 1024), caracacterEmplacementIsGrey=True,
                    caracterRecognitionRes = (32, 32), caracterRecognitionIsGrey = True):
    
    pass


def determineEmplacement(segmentationMap, threshold = 0.5):
    
    segmentationMap = np.where(segmentationMap>threshold, 1., 0.)
    segmentationMapLabel, numFeature = scipy.ndimage.label(segmentationMap)
    caracterPosition = scipy.ndimage.find_objects(input, max_label=0)
    
    return caracterPosition


def createFinalFile(database, outputFile, unicodeFile):
    
    pass

def createLabel(databaseCaracter, unicodeFile):
    
    pass #return a string corresponding to the label format of the output

def rencodeCaracter(value, unicodeDatabase):
    
    return unicodeDatabase['Unicode'].iloc[value]