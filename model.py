from tensorflow import keras as k
import tensorflow as tf
from training_utils import DeformableConv2D

def get_trans_img_encoder(img):
    cam_conv1 = DeformableConv2D(16, (3,3), padding = "same", activation = "relu")(img)
    cam_conv1 = DeformableConv2D(16, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv1 = k.layers.MaxPooling2D((2,2))(cam_conv1)
    
    cam_conv2 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv2 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv2 = k.layers.MaxPooling2D((2,2))(cam_conv2)
    
    cam_conv3 = DeformableConv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv3 = DeformableConv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv3 = k.layers.MaxPooling2D((2,2))(cam_conv3)
    
    cam_conv4 = DeformableConv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv4 = DeformableConv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv4)
    cam_conv4 = k.layers.MaxPooling2D((2,2))(cam_conv4)
    
    return cam_conv1, cam_conv2, cam_conv3, cam_conv4

def get_orig_img_encoder(img):
    conv1 = k.layers.SeparableConv2D(16, (3,3), padding = "same", activation = "relu")(img)
    conv1 = k.layers.SeparableConv2D(16, (3,3), padding = "same", activation = "relu")(conv1)
    conv1 = k.layers.MaxPooling2D((4,4))(conv1)
    
    conv2 = k.layers.SeparableConv2D(32, (3,3), padding = "same", activation = "relu")(conv1)
    conv2 = k.layers.SeparableConv2D(32, (3,3), padding = "same", activation = "relu")(conv2)
    conv2 = k.layers.MaxPooling2D((4,4))(conv2)
    
    return conv2


def get_model(input_shape, number_of_points):
    
    inp1 = k.Input(input_shape)
    inp2 = k.Input(input_shape)
    
    bn1 = k.layers.BatchNormalization()(inp1)
    bn2 = k.layers.BatchNormalization()(inp2)
    
    og_img = get_orig_img_encoder(bn1)
    trans_img1, trans_img2, trans_img3, trans_img4 = get_trans_img_encoder(bn2)
    
    conc = k.layers.Concatenate(axis = -1)([og_img, trans_img4])
    
    feature_map = DeformableConv2D(64, (3,3), padding = "same", activation = "relu")(conc)
    feature_map = DeformableConv2D(64, (3,3), padding = "same", activation = "relu")(feature_map)
    feature_map = k.layers.MaxPooling2D()(feature_map)
    
    probs = k.layers.Flatten()(feature_map)
    probs = k.layers.Dense(number_of_points*2, activation = "relu")(probs)
    probs = k.layers.Dense(number_of_points, activation = "sigmoid")(probs)
    probs = k.layers.Reshape((number_of_points,1))(probs)
    
    feature_map1 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(feature_map)
    feature_map1 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(feature_map1)
    feature_map1 = k.layers.MaxPooling2D()(feature_map1)
    
    feature_map2 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(feature_map1)
    feature_map2 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(feature_map2)
    feature_map2 = k.layers.MaxPooling2D()(feature_map2)
    
    locations = k.layers.Flatten()(feature_map2)
    locationsX = k.layers.Dense(number_of_points)(locations)
    locationsY = k.layers.Dense(number_of_points)(locations)
    
    locationsX = k.layers.Reshape((number_of_points,1))(locationsX)
    locationsY = k.layers.Reshape((number_of_points,1))(locationsY)
    output = k.layers.Concatenate(axis = -1)([probs, locationsX, locationsY])
    
    model = k.models.Model(inputs = [inp1, inp2], outputs = output)
    return model

# model = get_model((256,256,3), 5)
# model.summary()