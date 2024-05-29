import cv2
import numpy as np
import tensorflow.keras as k
from matplotlib import pyplot as plt

from training_utils import CustomDataGenerator

image_files = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/img/"
matrices_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/trasn/"
background_images_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/Background/"
original_image_path = "/Users/flam/Documents/SLAM/FlamImageTracking/test/img/cinema1.jpeg"
model_path = "/Users/flam/Documents/SLAM/FlamImageTracking/model.h5"
losses_file = "/Users/flam/Documents/SLAM/FlamImageTracking/train_loss.txt"
image_size = (256,256)
original_image = np.expand_dims(cv2.resize(cv2.imread(original_image_path), image_size), axis=0)
model = k.models.load_model(model_path)
data_generator = CustomDataGenerator(original_image_path, image_files, matrices_path, background_images_path,[[0.5,0.5]],1,400, None)
plt.ion()

def plot_losses(file_path, point, groundtruth, input_shape):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    probability = str(np.float16(point[0,0,0]))
    transformed_point = (int(point[0,0,1]*input_shape[2]), int(point[0,0,2]*input_shape[1]))
    GT_probability = str(np.float16(groundtruth[0,0,0]))
    GT_transformed_point = (int(groundtruth[0,0,1]*input_shape[2]), int(groundtruth[0,0,2]*input_shape[1]))
    
    name = f"""Predicted: probability: {probability}, point: {transformed_point}
    Groundtruth: probability: {GT_probability}, point: {GT_transformed_point}"""
    
    data = [float(line.strip()) for line in lines]
    plt.figure(figsize=(10, 6))
    plt.plot(data,color='b')
    plt.title(name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.draw()

# Function to display an image using OpenCV
def display_image(image, point):
    transformed_point = (int(point[0,0,1]*image[0].shape[1]), int(point[0,0,2]*image[0].shape[0]))
    cv2.circle(image[0], transformed_point, radius=3, color=(0,0,255), thickness=-1)
    image = np.array(image)
    
    # Position the OpenCV window
    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)
    cv2.moveWindow("inference", 500, 50)
    cv2.imshow("inference", image[0])

# Main loop to iterate through images
while True:
    model = k.models.load_model(model_path)
    inputs, ground_truth = data_generator.__getitem__(np.random.randint(0,data_generator.__len__()))
    transformed_point = model(inputs)

    plot_losses(losses_file, transformed_point, ground_truth, inputs[1].shape)
    display_image(inputs[1], transformed_point)
    # Wait for a key press
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Destroy all OpenCV windows
    cv2.destroyAllWindows()
    plt.close()
