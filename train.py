import tensorflow as tf
from model import get_model
from training_utils import *

Dataset = "Dataset"
img_path =  os.path.join(Dataset,"img")
mat_path =os.path.join(Dataset,"trasn")
background_images_path = os.path.join(Dataset,"Background")
original_image_path =os.path.join(os.path.join('test',"img"),"cinema1.jpeg")
keypoints = np.array([[0.5,0.5],[0,1],[1,0],[0,0],[1,1]])
dataset_size = 400
input_shape = (256,256,3)
epochs = 1000
batch_size = 10
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
seed = 12345


model = get_model(input_shape, len(keypoints), seed)
# model = tf.keras.models.load_model("model_backup.h5")
print(model.summary())

generator = CustomDataGenerator(original_image_path, img_path, mat_path, background_images_path, keypoints, batch_size, dataset_size, seed)

train(generator,model,epochs, optimizer)
