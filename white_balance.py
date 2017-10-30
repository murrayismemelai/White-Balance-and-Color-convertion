import cv2
import numpy as np
from cc import *
from matplotlib import pyplot as plt
img = cv2.imread('input1.bmp')
color = ('b','g','r')

def show_histr(img):
    histr = np.array([])
    #plt.figure(1)
    plt.subplot(2, 1, 1)
    for channel in range(img.shape[2]):
        accumulate = np.zeros((256,),dtype=np.int)
        for height in range(img.shape[0]):
            for width in range(img.shape[1]):
                accumulate[img[height][width][channel]] = accumulate[img[height][width][channel]] + 1
        plt.plot(accumulate,color = color[channel])
        plt.xlim([0,255])
        histr = np.append(histr,accumulate)
    #plt.figure(2)
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    histr = histr.reshape(3,-1)
    return histr
    
def gimp(img, perc = 0.05):
    for channel in range(img.shape[2]):
        mi, ma = (np.percentile(img[:,:,channel], perc), np.percentile(img[:,:,channel],100.0-perc))
        img[:,:,channel] = np.uint8(np.clip((img[:,:,channel]-mi)*255.0/(ma-mi), 0, 255))
    return img

out = img 
out = gray_world(out)
out = gimp(out,0.05)
cv2.imwrite('wb.bmp',out)

show_histr(out)
