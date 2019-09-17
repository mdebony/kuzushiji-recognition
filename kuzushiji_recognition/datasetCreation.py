import pandas as pd
import numpy as np
from PIL import Image
from .imageManipulation import *
from tqdm import tqdm

def createCaracDatabase(label, unicodeData):
    carac = []
    pos = []
    size = []
    if(not isinstance(label, float)):
        labelS = label.split()
        for i in range(0, len(labelS), 5):
            carac.append(unicodeData[unicodeData['Unicode']==labelS[i]]['char'].iloc[0])
            pos.append((int(labelS[i+1]),int(labelS[i+2])))
            size.append((int(labelS[i+3]),int(labelS[i+4])))
    
    return pd.DataFrame(list(zip(carac, pos, size)), columns=['caracter', 'position', 'size'])


def fromPageCreateCaracterDataset(image, label, unicodeData):
    caracDB = createCaracDatabase(label, unicodeData)
    imageCaracList = []
    caracList = []
    for i in range(0, len(caracDB)):
        caracList.append(caracDB.iloc[i]['caracter'])
        if (caracDB.iloc[i]['size'][0] > 0) and (caracDB.iloc[i]['size'][1] > 0):
            imArray = np.asarray(extractSImageFromImage(image, caracDB.iloc[i]['position'], caracDB.iloc[i]['size'])).copy()
            imArray = (np.sum(imArray, axis=2)/3.).astype(np.uint8)
            imageCaracList.append(imArray.tolist())
        else:
            print('error')
    return caracList, imageCaracList


def loadImageCaracterEmplacement(importFile, folder, caracterEmplacementRes = (1024, 1024), caracacterEmplacementIsGrey=True):
    
    pass

def loadImageCaracterRecognition(databaseCaracter, folder, caracterRecognitionRes = (32, 32), caracterRecognitionIsGrey = True):
    
    pass


def createDatasetFirstNetwork(inputFile, outputFile, unicodeData, imageRep='../data/train_images/', xpixel=1024, ypixel=1024, gray=True):
    data = pd.read_csv(inputFile)
#    unicodeData = pd.read_csv(unicodeFile)
    
    nbImage = data.shape[0]
    segMaps = np.zeros((nbImage, xpixel, ypixel, 2), dtype=np.uint8)
    cImages = np.zeros((nbImage, xpixel, ypixel, 1), dtype=np.uint8)
    
    print('Creating segmentation maps and converting train images', flush = True)
    for j in tqdm(range(0, nbImage)):
        idImage = data['image_id'].iloc[j]
        label = data['labels'].iloc[j]
        Im = Image.open(imageRep + idImage + '.jpg')
        xIm, yIm = Im.size
        dB = createCaracDatabase(label, unicodeData)
        segMap = createSegmentationMap(xIm, yIm, dB)
        segMapCenter = createSegmentationCenterMap(xIm, yIm, dB, (7,7))
        cSegMap = convertImage(Image.fromarray(segMap), xpixel, ypixel, gray=True, squared=True, squared_fill_color=(0, 0, 0))
        cSegMapCenter = convertImage(Image.fromarray(segMapCenter), xpixel, ypixel, gray=True, squared=True, squared_fill_color=(0, 0, 0), algorithmDownscale=Image.NEAREST)
        cSegMap = (np.sum(np.array(cSegMap), axis=2)/3.).astype(np.uint8)
        cSegMapCenter = (np.sum(np.array(cSegMapCenter), axis=2)/3.).astype(np.uint8)
        cSegMaps = np.swapaxes(np.array([cSegMap,cSegMapCenter],dtype=np.uint8), 0, 2)
        segMaps[j] = cSegMaps
        cImages[j,:,:,0] = (np.sum(np.asarray(convertImage(Im, xpixel, ypixel, gray=True, squared=True), dtype=np.uint8), axis=2)/3.).astype(np.uint8)
        
    np.savez_compressed(outputFile, maps = segMaps, compressed_images = cImages)
    
        

def createDatasetSecondNetwork(inputFile, outputFile, imageRep='../data/train_images/', unicodeFile = '../data/unicode_translation.csv'):
    data = pd.read_csv(inputFile)
    unicodeData = pd.read_csv(unicodeFile)
    
    nImage = data.shape[0]
    imageCaracList = []
    caracList = []
    
    print('Convert train image', flush=True)
    for j in tqdm(range(0, nImage)):
        idImage = data['image_id'].iloc[j]
        label = data['labels'].iloc[j]
        image = Image.open(imageRep+idImage+'.jpg')
        tmp1, tmp2 = fromPageCreateCaracterDataset(image, label, unicodeData)
        caracList += tmp1
        imageCaracList += tmp2
        
    caracList = np.asarray(caracList)
    imageCaracList = np.asarray(imageCaracList, dtype=np.uint8)
    
    print('Create character table', flush=True)
    charTable = np.zeros((imageCaracList.shape[0], len(unicodeData)), dtype=np.bool)
    charClass = np.zeros((imageCaracList.shape[0]), dtype=np.int16)
    for i in tqdm(range(imageCaracList.shape[0])):
        charClass[i] = unicodeData[unicodeData['char']==caracList[i]].index.values.astype(int)[0]
    
    np.savez_compressed(outputFile, character = caracList, characterClass = charClass, image = imageCaracList)
    del imageCaracList
    del caracList
    del charClass