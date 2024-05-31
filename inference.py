import cv2
import numpy as np
import tensorflow.keras as k
from matplotlib import pyplot as plt

from training_utils import CustomDataGenerator

image_files = "Dataset//img//"
matrices_path = "Dataset//trasn//"
background_images_path = "Dataset//Background//"
original_image_path = "test\img\cinema1.jpeg"
model_path = "model.h5"
losses_file = "train_loss.txt"
image_size = (256,256)
original_image = np.expand_dims(cv2.resize(cv2.imread(original_image_path), image_size), axis=0)
model = k.models.load_model(model_path)
data_generator = CustomDataGenerator(original_image_path, image_files, matrices_path, background_images_path,np.array([[0.5,0.5],[0,1],[1,0],[0,0],[1,1]]),1,400, None)
plt.ion()

colours = [[0,0,255],[0,255,0],[255,0,0],[255,255,0],[255,0,255],[0,255,255]]

def plot_losses(file_path, point, groundtruth, input_shape):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    probability = str(np.float16(point[0,:,0]))
    transformed_point = np.array((point[0,:,1]*input_shape[2], point[0,:,2]*input_shape[1]), dtype=np.int16)
    GT_probability = str(np.float16(groundtruth[0,:,0]))
    GT_transformed_point = np.array((groundtruth[0,:,1]*input_shape[2] ,groundtruth[0,:,2]*input_shape[1]), dtype = np.int16)
    
    name = f"""Predicted: probability: {probability},
    point: {transformed_point}
    Groundtruth: probability: {GT_probability},
    point: {GT_transformed_point}"""
    
    data = [float(line.strip()) for line in lines]
    plt.figure(figsize=(10, 6))
    plt.plot(data,color='b')
    plt.title(name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.grid(True)
    plt.draw()

# Function to display an image using OpenCV
def display_image(image, point):
    transformed_points = np.array((point[0,:,1]*image[0].shape[1], point[0,:,2]*image[0].shape[0]), dtype = np.int16)
    transformed_points = np.transpose(transformed_points)
    
    for i,transformed_point in enumerate(transformed_points):
        cv2.circle(image[0], transformed_point, radius=3, color=colours[i], thickness=-1)
    # image = np.array(image)
    
    # Position the OpenCV window
    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)
    cv2.moveWindow("inference", 500, 75)
    cv2.imshow("inference", image[0])

# Main loop to iterate through images
while True:
    model = k.models.load_model(model_path)
    inputs, ground_truths = data_generator.__getitem__(np.random.randint(0,data_generator.__len__()))
    transformed_points = model(inputs)

    inputs[0] = np.array(inputs[0]*255, dtype=np.uint8)
    inputs[1] = np.array(inputs[1]*255, dtype=np.uint8)

    
    plot_losses(losses_file, transformed_points, ground_truths, inputs[1].shape)
    display_image(inputs[1], transformed_points)
    # Wait for a key pressq
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Destroy all OpenCV windows
    cv2.destroyAllWindows()
    plt.close()
