from tensorflow import keras
import scipy


def analyseDocument(importFile, outputFile, folderImage, unicodeFile, modelCaracterEmplacement, modelCaracterRecognition,
                    caracterEmplacementRes = (1024, 1024), caracacterEmplacementIsGrey=True, caracterEmplacementBatchSize = 16,
                    caracterRecognitionRes = (32, 32), caracterRecognitionIsGrey = True, caracterRecognitionBatchSize = 256,
                    thresholdEmplacement = 0.5):
    
    print("Load model caracter emplacement")
    model = keras.models.load_model(modelCaracterEmplacement)
    
    print("Load dataset caracter emplacement")
    imageDataset, xInit, yInit, xThumb, yThumb, xShift, yShift = loadImageCaracterEmplacement(importFile,
                                                                                              folderImage, caracterEmplacementRes = caracterEmplacementRes,
                                                                                              caracacterEmplacementIsGrey=caracacterEmplacementIsGrey)
    imageDataset = (imageDataset/255).astype(np.float16)
    
    print("Predict caracter emplacement")
    segmentationMaps = model.predict(imageDataset = imageDataset, batch_size=caracterEmplacementBatchSize)
    del imageDataset
    del model
    
    print("Create emplacement database")
    database = createEmplacementDatabase(segmentationMaps, thresholdEmplacement)
    del segmentationMaps
    
    
    print("Load model caracter recognition")
    model = keras.models.load_model(modelCaracterRecognition)
    
    print("Load caracter dataset")
    imageDataset = loadImageCaracterRecognition(database, folderImage,
                                                caracterRecognitionRes = modelCaracterRecognition,
                                                caracterRecognitionIsGrey = caracterRecognitionIsGrey)
    imageDataset = (imageDataset/255).astype(np.float16)
    
    print("Predict caracter")
    caracterValue = model.predict(imageDataset = imageDataset, batch_size=caracterEmplacementBatchSize)
    del imageDataset
    del model
    
    
    print("Create output file")
    createFinalFile(database, outputFile, unicodeFile)
    
    
def createEmplacementDatabase(segmentationMaps, threshold = 0.5):

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