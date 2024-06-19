import tensorflow as tf

def down_conv(inputs, filters):
    x = tf.keras.layers.MaxPooling2D()(inputs)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(x)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(x)
    
    return x

def up_conv(map, skip_map, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, (2,2), strides = (2,2), padding = "same", activation = "relu")(map)
    x = tf.keras.layers.Concatenate(axis = -1)([skip_map, x])
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(x)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(x)
    return x
def up_conv_module(inputs, down_convs):
    
    upconv4 = up_conv(inputs, down_convs[-1], 128)
    upconv3 = up_conv(upconv4, down_convs[-2], 64)
    upconv2 = up_conv(upconv3, down_convs[-3], 32)
    upconv1 = up_conv(upconv2, down_convs[-4], 8)
    
    output = tf.keras.layers.Conv2D(1, (1,1), padding = "same", activation = "relu")(upconv1)
    return output
    
def get_model(input_shape, number_of_points):
    
    inp = tf.keras.layers.Input(input_shape)
    
    conv1 = tf.keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu")(inp)
    conv1 = tf.keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu")(conv1)
    
    conv2 = down_conv(conv1, 16)
    conv3 = down_conv(conv2, 32)
    conv4 = down_conv(conv3, 64)
    conv5 = down_conv(conv4, 128)
    
    upconv5 = tf.keras.layers.Conv2D(256, (3,3), padding = "same", activation = "relu")(conv5)
    
    feature_maps = []
    for i in range(number_of_points):
        feature_maps.append(up_conv_module(upconv5, [conv1, conv2, conv3, conv4]))
        
    output = tf.keras.layers.Concatenate(axis = -1)(feature_maps)
    
    model = tf.keras.models.Model(inputs = inp, outputs = output)
    
    return model
    
model = get_model((256,256,3), 5)
model.summary()
model.save("testing_model.h5")