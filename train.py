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
descriptor_length = Configurations["image_configs"]["descriptor_length"]

epochs = Configurations["training_configs"]["epochs"]
train_batch_size = Configurations["training_configs"]["train_batch_size"]
val_batch_size = Configurations["training_configs"]["val_batch_size"]
optimizer = tf.keras.optimizers.Adam(learning_rate = Configurations["training_configs"]["learning_rate"])
seed = Configurations["training_configs"]["seed"]


model = get_model(input_shape, descriptor_length)
# model = tf.keras.models.load_model("model_backup.h5")
print(model.summary())

train_generator = CustomDataGenerator(train_batch_size, seed)
val_generator = CustomDataGenerator(val_batch_size, seed)

train(train_generator, val_generator, model, epochs, optimizer)