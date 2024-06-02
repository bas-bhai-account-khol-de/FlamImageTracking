from tensorflow import keras as k
import tensorflow as tf

def image_encoder(img):
    
    x = k.layers.Conv2D(filters = 8, kernel_size = (3,3), padding = "same", activation = "relu")(img)
    x = k.layers.Conv2D(filters = 8, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.MaxPooling2D()(x)
    
    x = k.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.MaxPooling2D()(x)
    
    x = k.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.MaxPooling2D()(x)
    
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.MaxPooling2D()(x)

    return x

def model_body(img):
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(img)
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.MaxPooling2D()(x)
    
    return x
    
def get_probs(img, number_of_points):
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(img)
    x = k.layers.MaxPooling2D()(x)
    
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(units = number_of_points*2, activation = "relu")(x)
    x = k.layers.Dense(units = number_of_points, activation = "sigmoid")(x)
    x = k.layers.Reshape((number_of_points, 1))(x)
    
    return x    

def get_locations(img, number_of_points):
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(img)
    x = k.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu")(x)
    x = k.layers.MaxPooling2D()(x)
    
    location_x = k.layers.Flatten()(x)
    location_x = k.layers.Dense(units = number_of_points*2, activation = "relu")(location_x)
    location_x = k.layers.Dense(units = number_of_points, activation = "linear")(location_x)
    location_x = k.layers.Reshape((number_of_points, 1))(location_x)
    
    location_y = k.layers.Flatten()(x)
    location_y = k.layers.Dense(units = number_of_points*2, activation = "relu")(location_y)
    location_y = k.layers.Dense(units = number_of_points, activation = "linear")(location_y)
    location_y = k.layers.Reshape((number_of_points, 1))(location_y)
    
    return [location_x,location_y]

def get_model(input_shape, number_of_points):
    
    inp1 = k.Input(input_shape)
    inp2 = k.Input(input_shape)
    
    bn1 = k.layers.BatchNormalization()(inp1)
    bn2 = k.layers.BatchNormalization()(inp2)
    
    features_orig_img = image_encoder(bn1)
    features_tran_img = image_encoder(bn2)
    
    features = k.layers.Concatenate(axis = -1)([features_orig_img, features_tran_img])
    processed_features = model_body(features)
    
    probs = get_probs(processed_features, number_of_points)
    locations = get_locations(processed_features, number_of_points)
    output = k.layers.Concatenate(axis = -1)([probs, *locations])
    
    model = k.models.Model(inputs = [inp1, inp2], outputs = output)
    return model

model = get_model((256,256,3), 5)
model.summary()