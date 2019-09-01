import pandas as pd
from .imageManipulation import convertImage, extractSImageFromImage

def createCaracDatabase(unicodeData,label):
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


def fromPageCreateCaracterDataset(unicodeData,image, label):
    caracDB = createCaracDatabase(unicodeData,label)
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
        

def createDatasetSecondNetwork(inputFile, outputFile):
    data = pd.read_csv(inputFile)
    
    nImage = data.shape[0]
    imageCaracList = []
    caracList = []
    
    print('Convert train image')
    print('\n')
    for j in range(0, nImage):
        update_progress(float(j+1)/float(nImage))
        idImage = data['image_id'].iloc[j]
        label = data['labels'].iloc[j]
        image = Image.open(dataRep+'train_images/'+idImage+'.jpg')
        tmp1, tmp2 = fromPageCreateCaracterDataset(unicodeData,image, label)
        caracList += tmp1
        imageCaracList += tmp2
        
    caracList = np.asarray(caracList)
    imageCaracList = np.asarray(imageCaracList, dtype=np.uint8)
    np.savez_compressed(outputFile, caracter = caracList, image = imageCaracList)
    del imageCaracList
    del caracList