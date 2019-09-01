from PIL import Image
import numpy as np

def makeSquareImage(im, minSize=32, fill_color=(255, 255, 255)):
    x, y = im.size
    size = max(max(x, y), minSize)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def convertImage(image, xpixel=1024, ypixel=1024, gray=False, squared=True):
    #convert size
    image.thumbnail([xpixel, ypixel], Image.LANCZOS)
    image.convert('RGB')
    
    #make square
    if squared:
        image = makeSquareImage(image, minSize=max(xpixel, ypixel), fill_color=(255,255,255))
    
    #convert color
    if gray:
        image.convert('L')
        
    return image

def extractSImageFromImage(image, pos, size, finalSize=(32,32), gray=False):
    box = (pos[0], pos[1], pos[0]+size[0], pos[1]+size[1])
    return convertImage(image.copy().crop(box), xpixel=finalSize[0], ypixel=finalSize[0], gray=gray)