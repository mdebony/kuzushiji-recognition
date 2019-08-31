import pandas as pd
import numpy as np
from PIL import Image
from imageManipulation import convertImage, extractSImageFromImage
from progressBar import update_progress
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


def createDatasetFirstNetwork(xpixel=1024, ypixel=1024, gray=False):
    nImage = trainData.shape[0]
    
    i=0
    print('Convert train image')
    print('\n')
    for idImage in trainData['image_id']:
        i+=1
        update_progress(float(i)/float(nImage))
        image = Image.open(dataRep+'train_images/'+idImage+'.jpg')
        image = convertImage(image,xpixel, ypixel, gray)
        image.save(datasetRep+'train/'+idImage+'.jpg')
        

def createDatasetSecondNetwork(inputFile, outputFile, imageRep='../data/train_images/', unicodeFile = '../data/unicode_translation.csv'):
    data = pd.read_csv(inputFile)
    unicodeData = pd.read_csv(unicodeFile)
    
    nImage = data.shape[0]
    imageCaracList = []
    caracList = []
    
    print('Convert train image')
    for j in tqdm(range(0, nImage)):
        idImage = data['image_id'].iloc[j]
        label = data['labels'].iloc[j]
        image = Image.open(imageRep+idImage+'.jpg')
        tmp1, tmp2 = fromPageCreateCaracterDataset(image, label, unicodeData)
        caracList += tmp1
        imageCaracList += tmp2
        
    caracList = np.asarray(caracList)
    imageCaracList = np.asarray(imageCaracList, dtype=np.uint8)
    
    print('Create character table')
    charTable = np.zeros((imageCaracList.shape[0], len(unicodeData)), dtype=np.bool)    
    for i in tqdm(range(imageCaracList.shape[0])):
        charTable[i, unicodeData[unicodeData['char']==caracList[i]].index.values.astype(int)[0]] = True
    
    np.savez_compressed(outputFile, character = caracList, characterMap = charTable, image = imageCaracList)
    del imageCaracList
    del caracList
    del charTable