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

def InceptionBlock(x):
    y= k.layers.SeparableConv2D(64,(5,5),padding='same',)(x)
    y= k.layers.LeakyReLU()(y)
    x= k.layers.SeparableConv2D(64,(3,3),padding='same')(x)
    x= k.layers.LeakyReLU()(x)

    x= k.layers.concatenate([x,y])
    return x

def InputImageNetwork(input_shape = (128,128,3))-> k.Model:
    inp = k.Input(input_shape)

    x= InceptionBlock(inp)
    x= InceptionBlock(x)
    x= k.layers.MaxPool2D()(x)
    x= InceptionBlock(x)
    x= InceptionBlock(x)
    x= k.layers.MaxPool2D()(x)
    x= InceptionBlock(x)
    x= InceptionBlock(x)

    
    model  = k.Model(inputs =inp,outputs =x,name='input_network')
    return model

def DetectionHead(x,number_points):
    prob =  k.layers.SeparableConv2D(number_points*3,(3,3),padding='same')(x)
    prob = k.layers.LeakyReLU()(prob)
    prob= k.layers.MaxPool2D()(prob)
    prob =  k.layers.SeparableConv2D(number_points*2,(3,3),padding='same')(prob)
    prob = k.layers.LeakyReLU()(prob)
    prob = k.layers.Flatten()(prob)
    return prob

def transformedImageNetwork(number_points,inp_shape = (32,32,128),input_shape = (128,128,3))-> k.Model:
    inp = k.Input(input_shape)
    inp_tensor = k.Input(inp_shape)

    x= InceptionBlock(inp)
    x= InceptionBlock(x)
    x= k.layers.MaxPool2D()(x)
    x= InceptionBlock(x)
    x= InceptionBlock(x)
    x= k.layers.MaxPool2D()(x)
    x= k.layers.concatenate([x,inp_tensor])

    x= InceptionBlock(x)
    x= InceptionBlock(x)
    x= InceptionBlock(x)
    x= k.layers.MaxPool2D()(x)
    x= InceptionBlock(x)
    x= InceptionBlock(x)

    prob = DetectionHead(x,number_points)
    prob =k.layers.Dense(number_points,activation='sigmoid')(prob)
    prob = k.layers.Reshape((number_points,1))(prob)

    locationX =  DetectionHead(x,number_points)
    locationX=k.layers.Dense(number_points,activation='linear')(locationX)
    locationX = k.layers.Reshape((number_points,1),name='location_x')(locationX)

    locationY =  DetectionHead(x,number_points)
    locationY=k.layers.Dense(number_points,activation='linear')(locationY)
    locationY = k.layers.Reshape((number_points,1),name="location_y")(locationY)

    f = k.layers.concatenate([prob,locationX,locationY])

    
    

    
    model  = k.Model(inputs =[inp,inp_tensor],outputs =f,name='transformed_network')
    return model



def finalModel(number_points, input_shape=(128,128,3)):
   
    orig = k.Input(input_shape)
    trans = k.Input(input_shape)
    m1= InputImageNetwork()
    m1.trainable =True
    inp =  m1(orig)
    
    m2 = transformedImageNetwork(number_points)
    m2.trainable=True
    res= m2([trans,inp])
    model = k.Model(inputs=[orig,trans],outputs =res )
    return model

def test ():
    ip = (256,256,3)
    model = finalModel(1)
    # trandom_tensor_normal = tf.random.normal(shape=[1,256, 256, 3], mean=0.0, stddev=1.0)
    # model(trandom_tensor_normal)
    model.summary()
  

test()
