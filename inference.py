import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from training_utils import CustomDataGenerator, DeformableConv2D, DroppingLayer, ResizingLayer, ChannelNormalization, get_corresponding_points, patches_to_images
from configs import Configurations

image_files = Configurations["paths"]["transformed_images_path"]
matrices_path = Configurations["paths"]["transformation_matrices_path"]
background_images_path = Configurations["paths"]["background_images_path"]
original_image_path = Configurations["paths"]["original_image_path"]
model_path = Configurations["paths"]["model_path"]
train_losses_file = Configurations["paths"]["train_losses_path"]
val_losses_file = Configurations["paths"]["val_losses_path"]
image_size = Configurations["image_configs"]["image_size"][:2]


colours = Configurations["inference_configs"]["colours"]
inference_thresh = Configurations["inference_configs"]["inference_thresh"]

original_image = np.expand_dims(cv2.resize(cv2.imread(original_image_path), image_size), axis=0)
data_generator = CustomDataGenerator(1, None)

plt.ion()

def plot_losses(train_file_path, val_file_path, key_points_orig, key_points_trans):
    with open(train_file_path, 'r') as file:
        train_lines = file.readlines()
        
    with open(val_file_path, 'r') as file:
        val_lines = file.readlines()
    
    name = f"""Losses
    key_points_orig: {key_points_orig.T},
    key_points_trans: {key_points_trans.T}"""

    train_data = [float(line.strip()) for line in train_lines]
    val_data = [float(line.strip()) for line in val_lines]
    plt.figure(figsize=(10, 6))
    plt.plot(train_data,color='b', label = """Training loss""")
    plt.plot(val_data,color='r', label = """Val loss""")
    plt.title(name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)
    plt.draw()

# Function to display an image using OpenCV
def display_image(image, pred_key_points, points_to_be_shown, location, title):
    for i,transformed_point in enumerate(pred_key_points):
        cv2.circle(image[0], (int(transformed_point[1]), int(transformed_point[0])), radius=3, color=colours[i], thickness=-1)
    # image = np.array(image)
    
    # Position the OpenCV window
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title, *location)
    cv2.imshow(title, image[0])

def post_process_probs(key_points):
    key_points = key_points[:,:,:,:-1]
    key_points = patches_to_images(key_points)
    return key_points


# Main loop to iterate through images
while True:
    with tf.keras.utils.custom_object_scope({'DeformableConv2D': DeformableConv2D,
                                      'DroppingLayer': DroppingLayer,
                                      'ResizingLayer': ResizingLayer,
                                      'ChannelNormalization': ChannelNormalization}):
        model = tf.keras.models.load_model(model_path)
    inputs, ground_truths = data_generator.__getitem__(np.random.randint(0,data_generator.__len__()))
    
    inputs[0] = np.array(inputs[0]*255, dtype=np.uint8)
    inputs[1] = np.array(inputs[1]*255, dtype=np.uint8)
    
    key_points_orig, descriptors_orig = model(inputs[0])
    key_points_trans, descriptors_trans = model(inputs[1])
    
    key_points_orig = post_process_probs(key_points_orig)
    key_points_trans = post_process_probs(key_points_trans)
    
    sns.heatmap(key_points_orig[0], cbar=False)
    
    orig_points_to_be_shown = key_points_orig > inference_thresh
    trans_points_to_be_shown = key_points_trans > inference_thresh
    
    sorted_indices = np.argsort(key_points_orig, axis=None)[::-1]
    sorted_indices_coords = np.unravel_index(sorted_indices, key_points_orig.shape[1:])
    key_points_orig = np.array([sorted_indices_coords[0][:5], sorted_indices_coords[1][:5]])
    key_points_orig = np.transpose(key_points_orig)
    key_points_trans = get_corresponding_points(key_points_orig, descriptors_orig, descriptors_trans)
    
    plot_losses(train_losses_file, val_losses_file, key_points_orig, key_points_trans)
    display_image(inputs[0], key_points_orig, orig_points_to_be_shown, (400,60), "original_image")
    display_image(inputs[1], key_points_trans, trans_points_to_be_shown, (650,60), "transformed_image")
    # display_image(inputs[0], ground_truths[0,:,1:], np.ones_like(orig_points_to_be_shown), (500,240), "actual")
    # Wait for a key press
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Destroy all OpenCV windows
    cv2.destroyAllWindows()
    plt.close()
