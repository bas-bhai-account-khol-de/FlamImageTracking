import tensorflow as tf
from model import get_model
from training_utils import *

img_path = "Dataset//img//"
mat_path = "Dataset//trasn//"
background_images_path = "Dataset//Background//"
original_image_path = "test\img\cinema1.jpeg"
keypoints = np.array([[0.5,0.5],[0,1],[1,0],[0,0],[1,1]])
dataset_size = 400
input_shape = (256,256,3)
epochs = 1000
batch_size = 10
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
seed = 12345


model = get_model(input_shape, len(keypoints), seed)
# model = tf.keras.models.load_model("/Users/flam/Documents/SLAM/FlamImageTracking/model_loss_0.0034785967.h5")
print(model.summary())

generator = CustomDataGenerator(original_image_path, img_path, mat_path, background_images_path, keypoints, batch_size, dataset_size, seed)

train(generator,model,epochs, optimizer)
