from tensorflow import keras as k
import tensorflow as tf
# from training_utils import k.layers.SeparableConv2D

def get_img_encoder(img):
    cam_conv1 = k.layers.SeparableConv2D(16, (3,3), padding = "same", activation = "relu")(img)
    cam_conv1 = k.layers.SeparableConv2D(16, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv1 = k.layers.MaxPooling2D((2,2))(cam_conv1)
    
    cam_conv2 = k.layers.SeparableConv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv1)
    cam_conv2 = k.layers.SeparableConv2D(32, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv2 = k.layers.MaxPooling2D((2,2))(cam_conv2)
    
    cam_conv3 = k.layers.SeparableConv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv2)
    cam_conv3 = k.layers.SeparableConv2D(64, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv3 = k.layers.MaxPooling2D((2,2))(cam_conv3)
    
    cam_conv4 = k.layers.SeparableConv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv3)
    cam_conv4 = k.layers.SeparableConv2D(128, (3,3), padding = "same", activation = "relu")(cam_conv4)
    cam_conv4 = k.layers.MaxPooling2D((2,2))(cam_conv4)
    
    return cam_conv1, cam_conv2, cam_conv3, cam_conv4

def get_model(input_shape, number_of_points):
    
    inp1 = k.Input(input_shape)
    inp2 = k.Input(input_shape)
    
    bn1 = k.layers.BatchNormalization()(inp1)
    bn2 = k.layers.BatchNormalization()(inp2)
    
    _, _, _, og_img = get_img_encoder(bn1)
    trans_img1, trans_img2, trans_img3, trans_img4 = get_img_encoder(bn2)
    
    conc = k.layers.Concatenate(axis = -1)([og_img, trans_img4])
    up_feature4 = k.layers.SeparableConv2D(128, (3,3), padding = "same", activation = "relu")(conc)
    up_feature4 = k.layers.Conv2DTranspose(64, (3,3), strides = (2,2), padding = "same")(up_feature4)
    
    up_feature3 = k.layers.Concatenate(axis = -1)([up_feature4, trans_img3])
    up_feature3 = k.layers.SeparableConv2D(64, (3,3), padding = "same", activation = "relu")(up_feature3)
    up_feature3 = k.layers.Conv2DTranspose(32, (3,3), strides = (2,2), padding = "same")(up_feature3)
    
    up_feature2 = k.layers.Concatenate(axis = -1)([up_feature3, trans_img2])
    up_feature2 = k.layers.SeparableConv2D(32, (3,3), padding = "same", activation = "relu")(up_feature2)
    up_feature2 = k.layers.Conv2DTranspose(16, (3,3), strides = (2,2), padding = "same")(up_feature2)
    
    up_feature1 = k.layers.Concatenate(axis = -1)([up_feature2, trans_img1])
    up_feature1 = k.layers.SeparableConv2D(16, (3,3), padding = "same", activation = "relu")(up_feature1)
    up_feature1 = k.layers.Conv2DTranspose(8, (3,3), strides = (2,2), padding = "same")(up_feature1)
    
    locations = k.layers.SeparableConv2D(5, (3,3), padding = "same", activation = "sigmoid")(up_feature1)
    
    probs = k.layers.SeparableConv2D(128,(3,3),padding = "same", activation = "relu")(conc)
    probs = k.layers.SeparableConv2D(64,(3,3),padding = "same", activation = "relu")(probs)
    probs = k.layers.MaxPooling2D()(probs)
    probs = k.layers.Flatten()(probs)
    probs = k.layers.Dense(2*number_of_points, activation = "relu")(probs)
    probs = k.layers.Dense(number_of_points, activation = "sigmoid")(probs)
    
    
    model = k.models.Model(inputs = [inp1, inp2], outputs = [probs, locations])
    return model

# model = get_model((256,256,3), 5)
# model.summary()