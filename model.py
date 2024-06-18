import tensorflow as tf
from training_utils import DeformableConv2D, DroppingLayer, ResizingLayer, ChannelNormalization

def VGG_network(inp, filters):
     conv1 = tf.keras.layers.Conv2D(filters, (3,3), padding = "same")(inp)
     conv1 = tf.keras.layers.LeakyReLU()(conv1)
     conv1 = tf.keras.layers.Conv2D(filters, (3,3), padding = "same")(conv1)
     conv1 = tf.keras.layers.LeakyReLU()(conv1)
     conv1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
     
     conv2 = tf.keras.layers.Conv2D(2*filters, (3,3), padding = "same")(conv1)
     conv2 = tf.keras.layers.LeakyReLU()(conv2)
     conv2 = tf.keras.layers.Conv2D(2*filters, (3,3), padding = "same")(conv2)
     conv2 = tf.keras.layers.LeakyReLU()(conv2)
     conv2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
     
     conv3 = tf.keras.layers.Conv2D(4*filters, (3,3), padding = "same")(conv2)
     conv3 = tf.keras.layers.LeakyReLU()(conv3)
     conv3 = tf.keras.layers.Conv2D(4*filters, (3,3), padding = "same")(conv3)
     conv3 = tf.keras.layers.LeakyReLU()(conv3)
     conv3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)
     
     return conv3

def detection_head(inp, input_shape):
    out = tf.keras.layers.Conv2D(65, (3,3), padding = "same", activation = "softmax")(inp)
    return out

def descriptor_head(inp, input_shape, descriptor_size):
    out = tf.keras.layers.Conv2D(descriptor_size, (3,3), padding = "same", activation = "linear")(inp)
    out = ResizingLayer(input_shape[:2])(out)
    out = ChannelNormalization()(out)
    return out

def get_model(input_shape, descriptor_length):
    
    inp1 = tf.keras.layers.Input(input_shape)
    bn1 = tf.keras.layers.BatchNormalization()(inp1)
    orig_img = VGG_network(bn1, 16)
    orig_detect = detection_head(orig_img, input_shape)
    orig_descriptor = descriptor_head(orig_img, input_shape, descriptor_length)
    model = tf.keras.models.Model(inputs = inp1, outputs = [orig_detect, orig_descriptor])
    return model
    
model = get_model((256,256,3), 256)
model.summary()
model.save("testing_model.h5")