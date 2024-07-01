import glfw
from OpenGL.GL import *
import numpy as np
import glm as m
from PIL import Image
from Homographysolver import *
import utils 
import pickle


def image_to_gl(point):
    return(point[0] -0.5,0.5-point[1])
def gl_to_imge(point):
    return((point[0]+ 1)/2,(1 - point[1])/2)

def load_and_display_mainImageAndCenter(filename,point = (0.5,0.5)): # point from top left corner image coordinte system
    image = cv2.imread(filename)
    h,w,_ = image.shape
    center = (w * point[0],h*point[1])
    
    utils.drawdots(image,[center])
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

def load_and_display_trsnformed_image(filename,tras_filename,point): # point from top left corner image coordinte system
    if filename is not None:
        image = cv2.imread(filename)
        h,w,_ = image.shape
        
    mat =[]
    with open(tras_filename, 'rb') as f:
        mat = pickle.load(f)
    
    gl_point = image_to_gl(point)
    rotMat = m.mat4(mat)
    
    
    pos =m.vec4(gl_point[0],gl_point[1],0,1)
    pos = rotMat * pos
    pos =pos.to_list()
    point = gl_to_imge([pos[0]/pos[2],pos[1]/pos[2]])
    center = ( point[0],point[1])
    # utils.drawdots(image,[center])
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return center
    


# point = (0.5,0.5)
# load_and_display_mainImageAndCenter('/Users/flam/Documents/SLAM/FlamImageTracking/test/img/cinema1.jpeg',point)
# load_and_display_trsnformed_image('/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/img/output2.png','/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/trasn/output2.pkl',point)