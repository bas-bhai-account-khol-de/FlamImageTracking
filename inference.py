import cv2
import numpy as np
import tensorflow.keras as k
from matplotlib import pyplot as plt

image_files = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/img/"
original_image_path = "/Users/flam/Documents/SLAM/FlamImageTracking/test/img/cinema1.jpeg"
model_path = "/Users/flam/Documents/SLAM/FlamImageTracking/model.h5"
losses_file = "/Users/flam/Documents/SLAM/FlamImageTracking/train_loss.txt"
image_size = (256,256)
original_image = np.expand_dims(cv2.resize(cv2.imread(original_image_path), image_size), axis=0)
model = k.models.load_model(model_path)
plt.ion()

def plot_losses(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = [float(line.strip()) for line in lines]
    plt.figure(figsize=(10, 6))
    plt.plot(data,color='b')
    plt.title('Line Plot of Decimal Numbers')
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
    cv2.namedWindow("probability "+str(np.float16(point[0,0,0])) + str(transformed_point), cv2.WINDOW_NORMAL)
    cv2.moveWindow("probability "+str(np.float16(point[0,0,0])) + str(transformed_point), 500, 50)
    cv2.imshow("probability "+str(np.float16(point[0,0,0])) + str(transformed_point), image[0])

# Main loop to iterate through images
while True:
    model = k.models.load_model(model_path)
    random_integer = np.random.randint(0, 400)
    image_path = image_files + "output" + str(random_integer) + ".png"
    image = np.expand_dims(cv2.resize(cv2.imread(image_path), image_size), axis= 0)
    transformed_point = model([original_image, image])
    
    plot_losses(losses_file)
    display_image(image, transformed_point)
    # Wait for a key press
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Destroy all OpenCV windows
    cv2.destroyAllWindows()
    plt.close()
