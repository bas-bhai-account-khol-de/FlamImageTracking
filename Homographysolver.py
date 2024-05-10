
import cv2
from utils import resizeRatio,drawPoints,applyTransform,createTrasnform
import numpy as np
import random


finalwidth  = 1000
ratio   = 1;
print('proof of Assumption :  we can calculate the $ T $ only by determining the new location on features points in screent at $t=t_n$. Following is the proof of assumption')

def CalculateHomoGraphy(orignalPoints,finalPoints,transform):  
    orignal = np.array(orignalPoints).reshape(10,4).transpose()
    final = np.array(finalPoints).reshape(10,4).transpose()
    
    p = np.array([[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,-1,0]])
    print(orignal)
    
    
    A_pseudo_inv = np.linalg.pinv(orignal)

    
    pt = np.dot(final,A_pseudo_inv)
    t = np.dot(np.linalg.pinv(p),pt)
    print(pt)
    tf = np.array(transform).reshape(4,4)
    tf= tf.transpose()
    print(tf)
    # print(np.dot(pt,))
   
    pass

if __name__ == '__main__':
    CalculateHomoGraphy(
                        [-0.219,-0.039,-1,1,0.001,-0.153,-1,1,-0.022,-0.311,-1,1,-0.266,0.137,-1,1,-0.301,0.301,-1,1,-0.053,-0.116,-1,1,0.312,-0.104,-1,1,0.315,-0.151,-1,1,0.257,-0.205,-1,1,0.061,-0.229,-1,1]
                        ,
                        [-0.215,-0.041,-1,1,0,-0.152,-1,1,-0.021,-0.313,-1,1,-0.269,0.157,-1,1,-0.309,0.352,-1,1,-0.051,-0.118,-1,1,0.235,-0.088,-1,1,0.238,-0.131,-1,1,0.198,-0.182,-1,1,0.05,-0.222,-1,1]
                        ,
                        [0.8775825618903728,0,-0.479425538604203,0,0,1,0,0,0.479425538604203,0,0.8775825618903728,0,0,0,-1,1]
                        )