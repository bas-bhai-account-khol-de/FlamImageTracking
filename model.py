import tensorflow as tf
from training_utils import ResizingLayer, ChannelNormalization

def backbone(inp, filters):
     conv1 = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(inp)
     conv1 = tf.keras.layers.BatchNormalization()(conv1)
     conv1 = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(conv1)
     conv1 = tf.keras.layers.BatchNormalization()(conv1)
     conv1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)

     conv2 = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(conv1)
     conv2 = tf.keras.layers.BatchNormalization()(conv2)
     conv2 = tf.keras.layers.Conv2D(filters, (3,3), padding = "same", activation = "relu")(conv2)
     conv2 = tf.keras.layers.BatchNormalization()(conv2)
     conv2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
     
     conv3 = tf.keras.layers.Conv2D(2*filters, (3,3), padding = "same", activation = "relu")(conv2)
     conv3 = tf.keras.layers.BatchNormalization()(conv3)
     conv3 = tf.keras.layers.Conv2D(2*filters, (3,3), padding = "same", activation = "relu")(conv3)
     conv3 = tf.keras.layers.BatchNormalization()(conv3)
     conv3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)
     
     conv4 = tf.keras.layers.Conv2D(2*filters, (3,3), padding = "same", activation = "relu")(conv3)
     conv4 = tf.keras.layers.BatchNormalization()(conv4)
     conv4 = tf.keras.layers.Conv2D(2*filters, (3,3), padding = "same", activation = "relu")(conv4)
     conv4 = tf.keras.layers.BatchNormalization()(conv4)
     
     return conv4

def detection_head(inp, input_shape):
    out = tf.keras.layers.Conv2D(256, (3,3), padding = "same", activation = "relu")(inp)
    out = tf.keras.layers.Conv2D(65, (1,1), padding = "same", activation = "relu")(out)
    out = tf.keras.layers.Softmax()(out)
    return out

def descriptor_head(inp, input_shape, descriptor_size):
    intermediate = tf.keras.layers.Conv2D(256, (3,3), padding = "same", activation = "linear")(inp)
    intermediate = tf.keras.layers.Conv2D(descriptor_size, (1,1), padding = "same", activation = "linear")(intermediate)
    intermediate = ResizingLayer(input_shape[:2])(intermediate)
    out = ChannelNormalization()(intermediate)
    return intermediate, out

def get_model(input_shape, descriptor_length):
    
    inp1 = tf.keras.layers.Input(input_shape)
    bn1 = tf.keras.layers.BatchNormalization()(inp1)
    orig_img = backbone(bn1, 16)
    orig_detect = detection_head(orig_img, input_shape)
    orig_intermediate_descriptor, orig_descriptor = descriptor_head(orig_img, input_shape, descriptor_length)
    model = tf.keras.models.Model(inputs = inp1, outputs = [orig_detect, orig_intermediate_descriptor, orig_descriptor])
    return model
    
# model = get_model((128,128,1), 128)
# a = model(tf.random.uniform((1,128,128,1)))
# print(a.shape)
# model.summary()
# model.save("testing_model.h5")
