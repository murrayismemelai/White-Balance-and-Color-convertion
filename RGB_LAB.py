import cv2
import numpy as np
from cc import *
from matplotlib import pyplot as plt

def gamma_effect(NL_RGB):
    NL_RGB = NL_RGB/255.0
    return np.where(NL_RGB>0.04045,np.power((NL_RGB+0.055)/(1+0.055),2.4),NL_RGB/12.92) 

def rgb2xyz(LN_BGR):
    rows = LN_BGR.shape[0]
    cols = LN_BGR.shape[1]
    LN_RGB = LN_BGR[:,:,::-1].reshape(-1,3)
    REC709 = np.array([[0.412453,0.357580,0.180423],[0.212671,0.715160,0.072169],[0.019334,0.119193,0.950227]])
    return np.dot(LN_RGB,REC709.T).reshape(rows,cols,3)
"""
def rgb2xyz(LN_BGR):
    rows = LN_BGR.shape[0]
    cols = LN_BGR.shape[1]
    LN_RGB = LN_BGR[:,:,::-1].reshape(-1,3)
    REC709 = np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    return np.dot(LN_RGB,REC709.T).reshape(rows,cols,3)
"""    
def f_t(t):
    return np.where(t>0.008856,np.power(t,1/3.0),7.787*t+16/116)

def xyz2lab(XYZ):
    rows = XYZ.shape[0]
    cols = XYZ.shape[1]
    XYZ = XYZ.reshape(-1,3)
    return np.array([116*f_t(XYZ[:,1])-16,500*(f_t(XYZ[:,0]/0.9515)-f_t(XYZ[:,1])),200*(f_t(XYZ[:,1])-f_t(XYZ[:,2]/1.0886))]).T.reshape(rows,cols,3)

img = cv2.imread('input2.jpeg')
out1 = xyz2lab(rgb2xyz(gamma_effect(img)))

def lab2xyz(LAB):
    rows = LAB.shape[0]
    cols = LAB.shape[1]
    LAB = LAB.reshape(-1,3)
    var_Y = (LAB[:,0]+16)/116.008856
    var_X = var_Y + LAB[:,1]/500.0
    var_Z = var_Y - LAB[:,2]/200.0
    Y = np.where(var_Y>6/29.0,np.power(var_Y,3),(var_Y-16/116)*3*np.power(6/29.0,2))
    X = np.where(var_X>6/29.0,0.9515*np.power(var_X,3),(var_X-16/116)*3*np.power(6/29.0,2)*0.9515)
    Z = np.where(var_Z>6/29.0,1.0886*np.power(var_Z,3),(var_Z-16/116)*3*np.power(6/29.0,2)*1.0886)
    return np.array([X,Y,Z]).T.reshape(rows,cols,3)

def xyz2rgb(XYZ):
    rows = XYZ.shape[0]
    cols = XYZ.shape[1]
    XYZ = XYZ.reshape(-1,3)
    inv_REC709 = np.array([[3.240479,-1.537150,-0.498535],[-0.969256,1.875992,0.041556],[0.055648,-0.204043,1.057311]])
    return np.dot(XYZ,inv_REC709.T).reshape(rows,cols,3)

def gamma_corr(LN_RGB):
    NL_RGB = np.where(LN_RGB>0.00304,(1+0.055)*np.power(LN_RGB,1/2.4)-0.055,LN_RGB*12.92) 
    NL_RGB = np.where(NL_RGB<0.,0,NL_RGB)
    NL_BGR = NL_RGB[:,:,::-1]*255.0
    return NL_BGR

    
out3 = out1

out3[:,:,1] = np.where(((out3[:,:,1]<-5)),out3[:,:,1] + 20,out3[:,:,1])
#success for gress
out3[:,:,0] = np.where(((out1[:,:,1]<-5)),out3[:,:,0],out3[:,:,0]*0.75)

#blue can be more dark?
#out3[:,:,2] = np.where(out1[:,:,2]<-20,out1[:,:,2] - 10,out1[:,:,2])
#out3[:,:,2] = np.where(((out1[:,:,2]<-20)|(out1[:,:,1]>6.0)),out1[:,:,2] + 10,out1[:,:,2])
#out3[:,:,1] = np.where(out1[:,:,2]<-15,out1[:,:,1] + 20,out1[:,:,1])


out2 = gamma_corr(xyz2rgb(lab2xyz(out3)))

cv2.imwrite('trans.jpeg',out2)