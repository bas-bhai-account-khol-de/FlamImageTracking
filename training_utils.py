import tensorflow as tf
from tensorflow import keras as k
from PointLocator import load_and_display_trsnformed_image
import numpy as np
import cv2
import pickle
import math
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

class CustomDataGenerator(k.utils.Sequence):
    def __init__(self, original_image_path, images_path, matrices_path, keypoints, batch_size,  dataset_size, seed,image_size = (256, 256)):
        k.utils.set_random_seed(seed)
        np.random.seed(seed)
        
        self.original_img = np.array(cv2.resize(cv2.imread(original_image_path), image_size))
        self.image_paths = []
        self.matrices = []
        for i in range(dataset_size):
            self.image_paths.append(images_path + "output" + str(i) + ".png")
            self.matrices.append(matrices_path + "output" + str(i) + ".pkl")
        
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
        
    def __getitem__(self, idx):
        low = idx * self.batchsize
        high = min(low + self.batchsize, len(self.matrices))
        
        batch_image_paths = self.image_paths[low:high]
        batch_matrices = self.matrices[low:high]
        batch_images = [cv2.resize(cv2.imread(path), self.imagesize) for path in batch_image_paths]
        batch_orig_img = np.tile(self.original_img, (len(batch_images), 1, 1, 1))
        
        batch_keypoints = []
        for i in range(len(batch_matrices)):
            transformed_points = []
            for keypoint in self.keypoints:
                transformed_point = load_and_display_trsnformed_image(batch_image_paths[i], batch_matrices[i], keypoint)
                is_present = 1
                if (transformed_point[0] < 0) | (transformed_point[0] > 1) | (transformed_point[1] < 0) | (transformed_point[1] > 1):
                    is_present = 0
                transformed_points.append((is_present, *transformed_point))
            batch_keypoints.append(transformed_points)
            
        batch_images = np.array(batch_images)
        batch_keypoints = np.array(batch_keypoints)
        
        return [batch_orig_img,batch_images], batch_keypoints
    
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