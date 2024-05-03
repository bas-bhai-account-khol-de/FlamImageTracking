
import cv2
import glm as m
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def resizeRatio(img  ,W):
    W = int(W)
    h,w,_  = img.shape
    r = h/w 
    H = int(W*r)
    img =cv2.resize(img,(W,H))
    return img



def createTrasnform():
    
    roation = 15
    roation1 =0
    rotMatz =m.rotate(roation/2,m.vec3(0,0,1))
    rotMaty =m.rotate(roation/3,m.vec3(0,1,0))
    rotMatx =m.rotate(-0.01,m.vec3(1,0,0))

    tranx= m.translate(m.vec3(-(roation%3)/6,0,0))
    trany = m.translate(m.vec3(0,-2*(roation%3)/6,0))
    tranz =  m.translate(m.vec3(0,0,-5))

    pers = m.perspective(m.radians(50),1,0.01,40)
    
    # test = m.mat4([[-0.5,-0.5,0,1],[ 0.5, -0.5,0,1],[ 0.5, 0.5,0,1],[-0.5, 0.5,0,1]])
    # test=m.transpose(test)


    transmat =  pers  * tranz * rotMatx 
    # transmat =  rotMatx 
    test2 = transmat * m.vec4(0.5,0.5,0,1)
    # test = transmat * test
    # global roation
    # roation+=0.01
    # transmat = m.transpose(transmat)
    # transmat = m.mat4() 
    print(transmat.to_list())
    return transmat




def drawPoints(img ,points ):
     image = img
     h,w,_ =  img.shape
     for point in points:
         cv2.circle(image,(int(((point[0])) *w),int(((point[1])) *h)),6,(0,0,255),-1)
     return image



def applyTransform(image , points , trasnform):
     pointset = points[:4]
     p = m.mat4(points)
     final  =  trasnform * p
     
     
     print("--------")
     
     
     pointset =  final.to_list()
     
     pointset = [ [x[0]/x[2],x[1]/x[2]] for x in pointset]
     
     
     o_points = points[:4]
     o_points = [[x[0],x[1]] for x in o_points]
     homo = cv2.getPerspectiveTransform(np.float32(o_points), np.float32(pointset))
     print(homo)
     warped  = cv2.warpPerspective(image,homo,(image.shape[1],image.shape[0]))
     return warped
     
     
     
     
    

def convert_opencv_to_pil(opencv_image):
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    # Convert from a NumPy array to a PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

createTrasnform()