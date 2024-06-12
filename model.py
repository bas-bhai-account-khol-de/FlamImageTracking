from tensorflow import keras as k
import tensorflow as tf
from training_utils import DeformableConv2D

def get_orig_img_encoder(img):
    cam_conv1 = k.layers.Conv2D(16, (3,3), padding = "same", activation = "relu")(img)
    cam_conv1 = k.layers.Conv2D(16, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv1 = k.layers.MaxPool2D((2,2))(cam_conv1)
    
    cam_conv2 = k.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv2 = k.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv2 = k.layers.MaxPool2D((2,2))(cam_conv2)
    
    cam_conv3 = k.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv3 = k.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv3 = k.layers.MaxPool2D((2,2))(cam_conv3)
    
    cam_conv4 = k.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv4 = k.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv4)
    cam_conv4 = k.layers.MaxPool2D((2,2))(cam_conv4)
    
    
    return cam_conv1, cam_conv2, cam_conv3, cam_conv4

def get_trans_img_encoder(img):
    cam_conv1 = DeformableConv2D(16, (3,3), padding = "same", activation = "relu")(img)
    cam_conv1 = DeformableConv2D(16, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv1 = k.layers.MaxPool2D((2,2))(cam_conv1)
    
    cam_conv2 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv2 = DeformableConv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv2 = k.layers.MaxPool2D((2,2))(cam_conv2)
    
    cam_conv3 = DeformableConv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv3 = DeformableConv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv3 = k.layers.MaxPool2D((2,2))(cam_conv3)
    
    cam_conv4 = DeformableConv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv4 = DeformableConv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv4)
    cam_conv4 = k.layers.MaxPool2D((2,2))(cam_conv4)
    
    return cam_conv1, cam_conv2, cam_conv3, cam_conv4

def decoder(feature_map, number_of_points, final_activation):
    output = k.layers.Dense(2*number_of_points)(feature_map)
    output = k.layers.LeakyReLU(0.3)(output)
    output = k.layers.Dense(number_of_points, activation = final_activation)(output)
    output = k.layers.Reshape((number_of_points, 1))(output)
    return output

def get_model(input_shape, number_of_points):
    
    inp1 = k.Input(input_shape)
    inp2 = k.Input(input_shape)
    
    bn1 = inp1 #k.layers.BatchNormalization()(inp1)
    bn2 = inp2 #k.layers.BatchNormalization()(inp2)
    
    _, _, _, org_img = get_orig_img_encoder(bn1)
    trns_img1, trns_img2, trns_img3, trns_img4 = get_trans_img_encoder(bn2)
    
    conc = k.layers.Concatenate(axis = -1)([trns_img4, org_img])
    feature_map = k.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(conc)
    feature_map = k.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(feature_map)
    
    feature_map = k.layers.Flatten()(feature_map)
    
    probs = decoder(feature_map,number_of_points, "sigmoid")
    locationX = decoder(feature_map, number_of_points, "linear")
    locationY = decoder(feature_map, number_of_points, "linear")
    
    output = k.layers.Concatenate(axis = -1)([probs, locationX, locationY])
    
    model = k.models.Model(inputs = [inp1, inp2], outputs = output)
    return model

# model = get_model((256,256,3), 5)
# model.summary()