import tensorflow as tf
from tensorflow import keras as k
from PointLocator import load_and_display_trsnformed_image
import numpy as np
import cv2
import pickle
import math
import os
from PIL import Image, ImageChops
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

class CustomDataGenerator(k.utils.Sequence):
    def __init__(self, original_image_path, images_path, matrices_path, background_images_path, keypoints, batch_size,  dataset_size, seed,image_size = (256, 256)):
        if seed is not None:
            k.utils.set_random_seed(seed)
            np.random.seed(seed)
        
        self.original_img = np.array(cv2.resize(cv2.imread(original_image_path), image_size))
        self.image_paths = []
        self.matrices = []
        for i in range(dataset_size):
            self.image_paths.append(images_path + "output" + str(i) + ".png")
            self.matrices.append(matrices_path + "output" + str(i) + ".pkl")
        
        self.background_images_path = [background_images_path + path for path in os.listdir(background_images_path)]
        self.batchsize = batch_size
        self.imagesize = image_size
        self.keypoints = keypoints
        
    def __len__(self):
        return math.ceil(len(self.matrices)/self.batchsize)
    
    def on_epoch_end(self):
        indices = np.array(range(len(self.matrices)))
        np.random.shuffle(indices)
        self.image_paths = self.image_paths[indices]
        self.matrices = self.matrices[indices]
        
    def process_images(self,image, background_image):
    
        rgb_r, rgb_g, rgb_b = cv2.split(background_image)
        rgba_r, rgba_g, rgba_b, rgba_a = cv2.split(image)
        r = np.where(rgba_a == 255, rgba_r, rgb_r)
        g = np.where(rgba_a == 255, rgba_g, rgb_g)
        b = np.where(rgba_a == 255, rgba_b, rgb_b)
        
        # If the image is less than 15% of total image, we can say it has no point
        prob = 1
        if (np.sum(rgba_a!=0) < (0.07*rgba_a.shape[0]*rgba_a.shape[1])):
            prob = 0
            
        merged_image = cv2.merge((r,g,b))
        return (merged_image, prob)
        
    def __getitem__(self, idx):
        low = idx * self.batchsize
        high = min(low + self.batchsize, len(self.matrices))
        
        batch_image_paths = self.image_paths[low:high]
        if self.batchsize == 1:
            print(batch_image_paths[0])
        batch_matrices = self.matrices[low:high]
        batch_background_paths = np.random.choice(self.background_images_path,high-low)
        
        batch_images = [cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), self.imagesize) for path in batch_image_paths]
        batch_background_images = [cv2.resize(cv2.imread(path), self.imagesize) for path in batch_background_paths]
        batch_processed_images = []
        probabilities = []
        
        for i in range(len(batch_images)):
            batch_processed_images.append(self.process_images(batch_images[i], batch_background_images[i]))
            
        batch_processed_images, probabilities = [list(t) for t in zip(*batch_processed_images)]
        batch_orig_img = np.tile(self.original_img, (len(batch_images), 1, 1, 1))
        
        batch_keypoints = []
        for i in range(len(batch_matrices)):
            transformed_points = []
            for keypoint in self.keypoints:
                transformed_point = load_and_display_trsnformed_image(batch_image_paths[i], batch_matrices[i], keypoint)
                is_present = probabilities[i]
                if (transformed_point[0] < 0) | (transformed_point[0] > 1) | (transformed_point[1] < 0) | (transformed_point[1] > 1):
                    is_present = 0
                transformed_points.append((is_present, *transformed_point))
            batch_keypoints.append(transformed_points)
            
        batch_processed_images = np.array(batch_processed_images)
        batch_keypoints = np.array(batch_keypoints)
        
        return [batch_orig_img,batch_processed_images], batch_keypoints
    
def custom_loss(y_true, y_pred):
    probability_true, key_points_true = y_true[:,:,0], y_true[:,:,1:] 
    probability_pred, key_points_pred = y_pred[:,:,0], y_pred[:,:,1:]
    
 
    bce = k.losses.BinaryCrossentropy()
    bce_loss = bce(probability_true, probability_pred)


    probability_true = probability_true.reshape((probability_true.shape[:2]))
    
    euclidian_loss = k.backend.square(key_points_true - key_points_pred)
    euclidian_loss = k.backend.sum(euclidian_loss, axis = -1)
    euclidian_loss = k.backend.sqrt(euclidian_loss)


    euclidian_loss = probability_true * euclidian_loss
    euclidian_loss = k.backend.mean(euclidian_loss)
    
    print(bce_loss, euclidian_loss)
    return bce_loss + euclidian_loss

def train(generator, model, epochs, optimizer):
    with open("SLAM/FlamImageTracking/train_loss.txt",'w') as writer:
        writer.write('')
    for epoch in range(epochs):
        print(f"Epoch {epoch} starting...")
        for batch in range(generator.__len__()):
            x, y = generator.__getitem__(batch)
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = custom_loss(y, predictions)  
                with open("SLAM/FlamImageTracking/train_loss.txt",'a') as writer:
                    writer.write(str(loss.numpy()) + '\n')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            model.save("SLAM/FlamImageTracking/model.h5", save_format='h5')
            if batch%5==0:
                print(loss)
                

# img_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/img/"
# mat_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/trasn/"
# original_image_path = "/Users/flam/Documents/SLAM/FlamImageTracking/test/img/cinema1.jpeg"
# background_images_path = "/Users/flam/Documents/SLAM/FlamImageTracking/Dataset/Background/"
# keypoints = np.array([[0.5,0.5]])

# k = CustomDataGenerator(original_image_path, img_path, mat_path, background_images_path, keypoints, 1, 1, 12345)
# img = k.__getitem__(0)
# cv2.imshow("jhgfghj", img[0][1][0])
# cv2.waitKey(10000)