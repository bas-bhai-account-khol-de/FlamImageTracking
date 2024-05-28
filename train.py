import tensorflow as tf
from model import get_model
from training_utils import *

img_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/img/"
mat_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/trasn/"
original_image_path = "/Users/flam/Documents/SLAM/FlamImageTracking/test/img/cinema1.jpeg"
keypoints = np.array([[0.5,0.5]])
dataset_size = 400
input_shape = (256,256,3)
epochs = 200
batch_size = 40
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
seed = 12345


model = get_model(input_shape, len(keypoints), seed)
print(model.summary())
generator = CustomDataGenerator(original_image_path, img_path, mat_path, keypoints, batch_size, dataset_size, seed)

train(generator,model,epochs, optimizer)
