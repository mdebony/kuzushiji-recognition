from PIL import Image
import numpy as np

def makeSquareImage(im, minSize=32, fill_color=(255, 255, 255), shiftData=False):
    x, y = im.size
    size = max(max(x, y), minSize)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    
    if shiftData:
        return new_im, (int((size - x) / 2), int((size - y) / 2))
    return new_im

def convertImage(image, xpixel=1024, ypixel=1024, gray=False, squared=True, squared_fill_color=(255, 255, 255), conversionData=False):
    #convert size
    convData={}
    x, y = im.size
    convData['init_x']=x
    convData['init_y']=y
    image.thumbnail([xpixel, ypixel], Image.LANCZOS)
    x, y = im.size
    convData['thumb_x']=x
    convData['thumb_y']=y
    image.convert('RGB')
    
    #make square
    if squared:
        if conversionData:
            image, shift = makeSquareImage(image, minSize=max(xpixel, ypixel), fill_color=squared_fill_color, shiftData=conversionData)
            convData['shift_x']=shift[0]
            convData['shift_y']=shift[1]
        else:
            image = makeSquareImage(image, minSize=max(xpixel, ypixel), fill_color=squared_fill_color, shiftData=conversionData)
    
    #convert color
    if gray:
        image.convert('L')
    
    if conversionData:
        return image, convData
    return image

def extractSImageFromImage(image, pos, size, finalSize=(32,32), gray=False):
    box = (pos[0], pos[1], pos[0]+size[0], pos[1]+size[1])
    return convertImage(image.copy().crop(box), xpixel=finalSize[0], ypixel=finalSize[0], gray=gray)

def conversionToInitialPosition(x, y, xInit, yInit, xThumb, yThumb, xShift, yShift):
    x = x-xShift
    y = y-yShift
    x = int(x*xInit/xThumb)
    y = int(y*yInit/yThumb)
    return x, y

def conversionToThumbPosition(x, y, xInit, yInit, xThumb, yThumb, xShift, yShift):
    x = int(x*xThumb/xInit)
    y = int(y*yThumb/yInit)
    x = x+xShift
    y = y+yShift
    return x, y

def createSegmentationMap(xSize, ySize, charaDB):
    segMap = np.zeros((xSize, ySize), dtype=np.uint8)
    
    for i in range(len(charaDB)):
        segMap[charaDB.iloc[i]['position'][0]:(charaDB.iloc[i]['position'][0]+charaDB.iloc[i]['size'][0]),
               charaDB.iloc[i]['position'][1]:(charaDB.iloc[i]['position'][1]+charaDB.iloc[i]['size'][1])] = 255
    
    return segMap

def separateDatabase(charaDB):
    charaDB1 = charaDB.iloc[0::2]
    charaDB2 = charaDB.iloc[1::2]
    return charaDB1, charaDB2

def testSegMap(segMapTot, segMap1, segMap2):
    temp = segMap1 + segMap2
    for i in range(len(temp)):
        for j in range(len(temp[0])):
            if temp[i][j] > 255:
                temp[i][j] = 255
    if (temp - segMapTot == np.zeros((len(temp), len(temp[0])), dtype=np.uint8)).all():
        print('Segmentation map correctly created')
    else:
        print('Segmentation map incorrectly created')