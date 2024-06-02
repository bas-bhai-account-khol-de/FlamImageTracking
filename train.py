import tensorflow as tf
from model import *
from training_utils import *
from configs import Configurations


img_path = Configurations["paths"]["transformed_images_path"]
mat_path = Configurations["paths"]["transformation_matrices_path"]
background_images_path = Configurations["paths"]["background_images_path"]
original_image_path = Configurations["paths"]["original_image_path"]

keypoints = keypoints = Configurations["image_configs"]["key_points"]
input_shape = Configurations["image_configs"]["image_size"]

epochs = Configurations["training_configs"]["epochs"]
batch_size = Configurations["training_configs"]["batch_size"]
optimizer = tf.keras.optimizers.Adam(learning_rate = Configurations["training_configs"]["learning_rate"])
seed = Configurations["training_configs"]["seed"]


model = get_model(input_shape, len(keypoints))
# model = tf.keras.models.load_model("model_backup.h5")
print(model.summary())

generator = CustomDataGenerator(batch_size, seed)

train(generator,model,epochs, optimizer)