
import cv2
from utils import resizeRatio,drawPoints,applyTransform,createTrasnform
import numpy as np
import random


finalwidth  = 1000
ratio   = 1;
print('proof of Assumption :  we can calculate the $ T $ only by determining the new location on features points in screent at $t=t_n$. Following is the proof of assumption')

def getMarkedIMageandPoints():
    #parameters 

    global finalwidth
    finalwidth  = 1000
    global ratio
    ratio   = 1;

    orig_image  =  cv2.imread("test/img/cinema1.jpeg")

    # resize to a particular size
    res_orig_image  =  resizeRatio(orig_image,finalwidth)
    ratio  =  res_orig_image.shape[0]/res_orig_image.shape[1]

    points  =  [ ([random.random(),random.random(),0,1]) for i in range(8)]

    print(points)

    marked_original_image =  drawPoints(res_orig_image,points)

    # trasnform = createTrasnform()
    # fn =applyTransform(resizeRatio(orig_image,finalwidth),points,trasnform)


    cv2.imshow('orignal image',marked_original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resizeRatio(orig_image,finalwidth)