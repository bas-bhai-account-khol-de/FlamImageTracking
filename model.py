from tensorflow import keras as k


def get_conv_block(inputs,filters, kernal_size, custom_padding = "same", custom_stride = (1, 1), custom_activation = "relu", custom_max_pool_kernel = (2,2)):
    conv1 = k.layers.Conv2D(filters, kernal_size, padding = custom_padding, strides = custom_stride, activation = custom_activation)(inputs)
    conv2 = k.layers.Conv2D(filters, kernal_size, padding = custom_padding, strides = custom_stride, activation = custom_activation)(conv1)
    max1 = k.layers.MaxPooling2D(custom_max_pool_kernel)(conv2)
    
    return max1
    

def get_model(input_shape, num_points,seed):
    k.utils.set_random_seed(seed)
    
    input_1 = k.Input(input_shape)
    input_2 = k.Input(input_shape)

    conv_block_1 = get_conv_block(input_1, filters = 32, kernal_size=(3,3), custom_max_pool_kernel = (4,4))
    conv_block_2 = get_conv_block(conv_block_1, filters = 64, kernal_size=(3,3), custom_max_pool_kernel = (4,4))

    cam_conv_block_1 = get_conv_block(input_2, filters = 32, kernal_size=(3,3))
    cam_conv_block_2 = get_conv_block(cam_conv_block_1, filters = 32, kernal_size=(3,3))
    cam_conv_block_3 = get_conv_block(cam_conv_block_2, filters = 64, kernal_size=(3,3))
    cam_conv_block_4 = get_conv_block(cam_conv_block_3, filters = 64, kernal_size=(3,3))

    combined_feature_vector = k.layers.Concatenate()([conv_block_2,cam_conv_block_4])
    
    feature_map1 = get_conv_block(combined_feature_vector, 32, (3,3), custom_max_pool_kernel = (4,4))
    feature_map2 = get_conv_block(feature_map1, 4, (3,3), custom_max_pool_kernel = (4,4))
    
    flattened_layer = k.layers.Flatten()(feature_map2)
    
    probabilities = k.layers.Dense(num_points, activation = "sigmoid")(flattened_layer)
    probabilities = k.layers.Reshape((num_points,1))(probabilities)
    locations = k.layers.Dense(2*num_points,activation = "linear")(flattened_layer)
    locations = k.layers.Reshape((num_points,2))(locations)
    
    output = k.layers.Concatenate(axis = -1)([probabilities,locations])
    
    model = k.models.Model(inputs = [input_1, input_2], outputs = output)
    return model