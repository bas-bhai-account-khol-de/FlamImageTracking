import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

from training_utils import CustomDataGenerator , process_GT
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

original_image = np.expand_dims(cv2.resize(cv2.imread(original_image_path), image_size), axis=0)
data_generator = CustomDataGenerator(1, None)

plt.ion()

def plot_losses(train_file_path, val_file_path, point, groundtruth, input_shape):
    with open(train_file_path, 'r') as file:
        train_lines = file.readlines()
        
    with open(val_file_path, 'r') as file:
        val_lines = file.readlines()
    
    probability = str(np.float16(point[0,:,0]))
    transformed_point = np.array((point[0,:,1], point[0,:,2]), dtype=np.int16)
    GT_probability = str(np.float16(groundtruth[0,:,0]))
    GT_transformed_point = np.array((groundtruth[0,:,1]*(input_shape[2]-1) ,groundtruth[0,:,2]*(input_shape[1]-1)), dtype = np.int16)
    
    name = f"""Predicted: probability: {probability},
    point: {transformed_point}
    Groundtruth: probability: {GT_probability},
    point: {GT_transformed_point}"""
    
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
def display_image(image, point):
    transformed_points = np.array((point[0,:,1], point[0,:,2]), dtype = np.int16)
    transformed_points = np.transpose(transformed_points)
    
    for i,transformed_point in enumerate(transformed_points):
        cv2.circle(image[0], transformed_point, radius=3, color=colours[i], thickness=-1)
    # image = np.array(image)
    
    # Position the OpenCV window
    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)
    cv2.moveWindow("inference", 500, 75)
    cv2.imshow("inference", image[0])

def display_heatmaps(ground_truths, transformed_points):
    plt.close()
    gt_transformed = process_GT(ground_truths[:,:,0],ground_truths[:,:,1:])
    fig, ax = plt.subplots(nrows=3, ncols=transformed_points.shape[-1], figsize = (15,8))
    for i in range(transformed_points.shape[-1]):
        small = np.min(gt_transformed[0,:,:,i])
        big = np.max(gt_transformed[0,:,:,i])
        sns.heatmap(gt_transformed[0,:,:,i], cbar=False, cmap="jet", ax = ax[0][i], vmin=0, vmax=max(1, big))
        ax[0][i].title.set_text(f"GT , min = {np.round(small, 2)}, max = {np.round(big, 2)}")
        
        small = np.min(transformed_points[0,:,:,i])
        big = np.max(transformed_points[0,:,:,i])
        ax[1][i].title.set_text(f"Pred, min = {np.round(small, 2)}, max = {np.round(big, 2)}")
        sns.heatmap(transformed_points[0,:,:,i], cbar=False, cmap="jet", ax = ax[1][i], vmin=0, vmax=max(1, big))
        
        small = (np.min(abs(gt_transformed[0,:,:,i] - transformed_points[0,:,:,i])))
        big = np.max(abs(gt_transformed[0,:,:,i] - transformed_points[0,:,:,i]))
        ax[2][i].title.set_text(f"Diff, min = {np.round(small, 2)}, max = {np.round(big, 2)}")
        sns.heatmap(abs(gt_transformed[0,:,:,i] - transformed_points[0,:,:,i]), cbar=False, cmap="jet", ax = ax[2][i], vmin=0, vmax=max(1, big))
        
    plt.tight_layout()
# Main loop to iterate through images
def featuremaps_to_arrays(map):
    arr = np.zeros((map.shape[0], map.shape[-1], 3))
    for img in range(map.shape[0]):
        for kp in range(map.shape[-1]):
            indices = np.argmax(map[img,:,:,kp])
            indices = np.unravel_index(indices, map.shape[1:3])
            arr[img, kp, :] = map[img, indices[0], indices[1], kp], indices[1], indices[0]
    return arr
    
while True:
    model = tf.keras.models.load_model(model_path)
    inputs, ground_truths = data_generator.__getitem__(np.random.randint(0,data_generator.__len__()))
    transformed_points = model(inputs[1])
    
    transformed_points_as_array = featuremaps_to_arrays(transformed_points)

    inputs[0] = np.array(inputs[0]*255, dtype=np.uint8)
    inputs[1] = np.array(inputs[1]*255, dtype=np.uint8)
    display_heatmaps(ground_truths, transformed_points)
    plot_losses(train_losses_file, val_losses_file, transformed_points_as_array, ground_truths, inputs[1].shape)
    display_image(inputs[1], transformed_points_as_array)
    # Wait for a key press
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Destroy all OpenCV windows
    cv2.destroyAllWindows()
    plt.close()