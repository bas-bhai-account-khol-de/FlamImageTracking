from tensorflow import keras as k
import tensorflow as tf

def get_conv_block(inputs,filters, kernal_size, custom_padding = "same", custom_stride = (1, 1), custom_activation = "linear", custom_max_pool_kernel = (2,2)):
    conv1 = k.layers.Conv2D(filters, kernal_size, padding = custom_padding, strides = custom_stride, activation = custom_activation)(inputs)
    conv1 = k.layers.LeakyReLU(0.3)(conv1)
    conv2 = k.layers.Conv2D(filters, kernal_size, padding = custom_padding, strides = custom_stride, activation = custom_activation)(conv1)
    conv2 = k.layers.LeakyReLU(0.3)(conv2)
    max1 = k.layers.MaxPooling2D(custom_max_pool_kernel)(conv2)
    
    return max1

def get_ASPP_block(inputs):
    return inputs

def get_model(input_shape, num_points,seed):
    
    filters = num_points *6
    # k.utils.set_random_seed(seed)
    
    input_1 = k.Input(input_shape)
    input_2 = k.Input(input_shape)

    bn1 = k.layers.BatchNormalization()(input_1)
    bn2 = k.layers.BatchNormalization()(input_2)
    
    conv_block_1 = get_conv_block(bn1, filters = filters, kernal_size=(3,3), custom_max_pool_kernel = (4,4))
    conv_block_2 = get_conv_block(conv_block_1, filters = filters*2, kernal_size=(3,3), custom_max_pool_kernel = (4,4))
    conv_block_3 = k.layers.Conv2D(64, (3,3), padding = "same", activation = "linear")(conv_block_2)
    conv_block_3 = k.layers.LeakyReLU(0.3)(conv_block_3)
    

    cam_conv_block_1 = get_conv_block(bn2, filters = filters, kernal_size=(3,3))
    cam_conv_block_2 = get_conv_block(cam_conv_block_1, filters = filters, kernal_size=(3,3))
    cam_conv_block_3 = get_conv_block(cam_conv_block_2, filters = filters*2, kernal_size=(3,3))
    cam_conv_block_4 = get_conv_block(cam_conv_block_3, filters = filters*2, kernal_size=(3,3))
    cam_conv_block_5 = k.layers.Conv2D(64, (3,3), padding = "same", activation = "linear")(cam_conv_block_4)
    cam_conv_block_5 = k.layers.LeakyReLU(0.3)(cam_conv_block_5)

    combined_feature_vector = k.layers.Concatenate()([conv_block_3,cam_conv_block_5])
    
    feature_map1 = get_conv_block(combined_feature_vector, filters, (3,3), custom_max_pool_kernel = (4,4))
    feature_map2 = get_conv_block(feature_map1, filters*2, (3,3), custom_max_pool_kernel = (4,4))
    feature_map3 = get_conv_block(feature_map2,filters , (3,3), custom_max_pool_kernel = (1,1))
    
    
    
    # flattened_layer = k.layers.Flatten()(feature_map3)
    
    # probabilities = k.layers.Dense(num_points*2, activation = "linear")(flattened_layer)
    # probabilities = k.layers.LeakyReLU(0.3)(flattened_layer)
    # probabilities = k.layers.Dense(num_points, activation = "sigmoid")(probabilities)
    # probabilities = k.layers.Reshape((num_points,1))(probabilities)
    
    # # locations = k.layers.Dense(2*num_points,activation = k.layers.LeakyReLU(0.3))(flattened_layer)
    # locations_x = k.layers.Dense(num_points,activation = "linear")(flattened_layer)
    # locations_y = k.layers.Dense(num_points,activation = "linear")(flattened_layer)
    
    # locations_x = k.layers.Reshape((num_points,1))(locations_x)
    # locations_y = k.layers.Reshape((num_points,1))(locations_y)
    # locations = k.layers.Reshape((num_points,2))(locations)
    
    feature_map4 = k.layers.Concatenate(axis = -1)([feature_map3,feature_map3,feature_map3])
    feature_map5 = k.layers.Conv2D(num_points*3*6, (3,3), activation = "linear", padding = "same")(feature_map4)
    feature_map5 = k.layers.LeakyReLU(0.3)(feature_map5)
    
    
    probabilities = k.layers.Dense(num_points*2, activation = "linear")(feature_map5)
    probabilities = k.layers.LeakyReLU(0.3)(probabilities)
    probabilities = k.layers.Dense(num_points, activation = "sigmoid")(probabilities)
    probabilities = k.layers.Reshape((num_points,1))(probabilities)
    
    # locations = k.layers.Dense(2*num_points,activation = k.layers.LeakyReLU(0.3))(flattened_layer)
    locations_x = k.layers.Dense(num_points,activation = "linear")(feature_map5)
    locations_y = k.layers.Dense(num_points,activation = "linear")(feature_map5)
    
    locations_x = k.layers.Reshape((num_points,1))(locations_x)
    locations_y = k.layers.Reshape((num_points,1))(locations_y)
    
    output = k.layers.Concatenate(axis = -1)([probabilities,locations_x,locations_y])
    
    
    model = k.models.Model(inputs = [input_1, input_2], outputs = output)
    return model


def test ():
    ip = (256,256,3)
    model = get_model(ip,1,10)
    trandom_tensor_normal = tf.random.normal(shape=[1,256, 256, 3], mean=0.0, stddev=1.0)
    model(trandom_tensor_normal)
    model.summary()

test()
