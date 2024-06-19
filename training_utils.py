import tensorflow as tf
from PointLocator import load_and_display_trsnformed_image
import numpy as np
import cv2
import math
import os
import warnings
from configs import Configurations

tf.experimental.numpy.experimental_enable_numpy_behavior()
warnings.simplefilter('ignore', category=FutureWarning)

image_size = Configurations["image_configs"]["image_size"][:2]
keypoints = Configurations["image_configs"]["key_points"]

val_epoch_threshold = Configurations["training_configs"]["val_epoch_threshold"]
val_drop_threshold = Configurations["training_configs"]["val_drop_threshold"]
weight = Configurations["training_configs"]["lambda"]
mp = Configurations["training_configs"]["mp"]
mn = Configurations["training_configs"]["mn"]

original_image_path = Configurations["paths"]["original_image_path"]
images_path = Configurations["paths"]["transformed_images_path"]
matrices_path = Configurations["paths"]["transformation_matrices_path"]
background_images_path = Configurations["paths"]["background_images_path"]
loss_variation_file_path = Configurations["paths"]["loss_variation_path"]
train_loss_file_path = Configurations["paths"]["train_losses_path"]
val_loss_file_path = Configurations["paths"]["val_losses_path"]
model_path = Configurations["paths"]["model_path"]
best_model_path = Configurations["paths"]["best_model_path"]
backup_model_path = Configurations["paths"]["backup_model_path"]


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, seed):
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
            np.random.seed(seed)
        
        self.original_img = np.array(cv2.resize(cv2.imread(original_image_path), image_size))
        self.image_paths = []
        self.matrices = []
        for file in os.listdir(images_path):
            if not file.endswith(".png"):
                continue
            file_name = file[:-4]
            self.image_paths.append(os.path.join(images_path, file_name+".png"))
            self.matrices.append(os.path.join(matrices_path,file_name+".pkl"))
        
        self.background_images_path = []
        for path in os.listdir(background_images_path):
            if not path.endswith(".jpg"):
                continue
            total_path = os.path.join(background_images_path,path)
            self.background_images_path.append(total_path)
            
        self.batchsize = batch_size
        self.imagesize = image_size
        self.keypoints = keypoints
        self.indices=np.arange(0,len(self.matrices))
        
    def __len__(self):
        return math.ceil(len(self.matrices)/self.batchsize)
    

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
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
        
        index_sec = self.indices[low:high]
        batch_image_paths = [self.image_paths[i] for i in index_sec]
        if self.batchsize == 1:
            print(batch_image_paths[0])
        batch_matrices = [self.matrices[i] for i in index_sec]
        batch_background_paths = np.random.choice(self.background_images_path,high-low)
        
        batch_images = [cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), self.imagesize) for path in batch_image_paths]
        batch_background_images = [cv2.resize(cv2.imread(path), self.imagesize) for path in batch_background_paths]
        batch_processed_images = []
        probabilities = []
        
        ## Add duplicate images without anything to add
        
        for i in range(len(batch_images)):
            batch_processed_images.append(self.process_images(batch_images[i], batch_background_images[i]))
            
        batch_processed_images, probabilities = [list(t) for t in zip(*batch_processed_images)]
        
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
        
        ### Create duplicates without points
        if self.batchsize == 1:
            number_duplicates = 1
        else:
            number_duplicates = np.random.randint(int(self.batchsize*0.5),int(self.batchsize*1.5))
        batch_orig_img = np.tile(self.original_img, (len(batch_images) + number_duplicates, 1, 1, 1))
        batch_background_paths_duplicates = np.random.choice(self.background_images_path,number_duplicates)
        batch_background_images_duplicates = [cv2.resize(cv2.imread(path), self.imagesize) for path in batch_background_paths_duplicates]
        batch_processed_images = np.concatenate([batch_processed_images, np.array(batch_background_images_duplicates)], axis = 0)
        batch_keypoints = np.concatenate([batch_keypoints, np.zeros((number_duplicates,*(batch_keypoints.shape[1:])))])
        
        if self.batchsize == 1:
            random_number = np.random.rand()
            if random_number > 0.5:
                batch_orig_img = batch_orig_img[1:]
                batch_processed_images = batch_processed_images[1:]
                batch_keypoints = batch_keypoints[1:]
            else:
                batch_orig_img = batch_orig_img[:1]
                batch_processed_images = batch_processed_images[:1]
                batch_keypoints = batch_keypoints[:1]

        return [(batch_orig_img).astype(np.float32)/255.0, batch_processed_images.astype(np.float32)/255.0], batch_keypoints

def process_GT(probability_true, key_points_true):
    GT = np.zeros((probability_true.shape[0], *image_size, len(keypoints)))
    for img in range(probability_true.shape[0]):
        for kp in range(probability_true[img].shape[0]):
            if probability_true[img][kp]:
                x = key_points_true[img][kp][1]
                y = key_points_true[img][kp][0]
                x = int(x * (image_size[1]-1))
                y = int(y * (image_size[0]-1))
                gt = np.zeros((image_size[0], image_size[1]))
                max_dist = 0
                if (((x - 0)**2) + ((y - 0)**2)) > (max_dist**2):
                    max_dist = np.sqrt(((x - 0)**2) + ((y - 0)**2))
                
                if (((x - image_size[0])**2) + ((y - 0)**2)) > (max_dist**2):
                    max_dist = np.sqrt(((x - image_size[0])**2) + ((y - 0)**2))
                
                if (((x - image_size[0])**2) + ((y - image_size[1])**2)) > (max_dist**2):
                    max_dist = np.sqrt(((x - image_size[0])**2) + ((y - image_size[1])**2))
                
                if (((x - 0)**2) + ((y - image_size[1])**2)) > (max_dist**2):
                    max_dist = np.sqrt(((x - 0)**2) + ((y - image_size[1])**2))
                
                for i in range(image_size[0]):
                    for j in range(image_size[1]):
                        dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                        gt[i][j] = (max_dist - dist) / max_dist
                GT[img, :, :, kp] = gt
                
    return GT
    
def custom_loss(y_true, y_pred, is_val = False):
    probability_true, key_points_true = y_true[:,:,0], y_true[:,:,1:] 
    GT = process_GT(probability_true, key_points_true)
    msel = tf.keras.losses.MeanSquaredError ()
    
    return msel(GT, y_pred)
                    
def train(train_generator:CustomDataGenerator, val_generator:CustomDataGenerator, model, epochs, optimizer):
    with open(train_loss_file_path,'w') as writer:
        writer.write('')
    with open(val_loss_file_path,'w') as writer:
        writer.write('')
    with open(loss_variation_file_path,'w') as writer:
            writer.write('')
    
    
    for epoch in range(epochs):
        epoch_train_loss = np.float16(0)
        epoch_val_loss = np.float16(0)
        epoch_val_loss_min = np.inf
        
        print(f"Epoch {epoch} starting...")
        for batch in range(train_generator.__len__()):
            x, y = train_generator.__getitem__(batch)
            with tf.GradientTape() as tape:
                predictions_trans = model(x[1])
                loss = custom_loss(y, predictions_trans)
                epoch_train_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            try:
                model.save(model_path)
                model.save(backup_model_path)
            except Exception as e:
                print("========= Note ================")
                print("cant save last model")
                print(e)
            
            
            # ### ======= Validation loop ============
            # val_x, val_y = val_generator.__getitem__(np.random.randint(0,val_generator.__len__()))
            # val_predictions_trans = model(val_x[1])
            # val_loss = custom_loss(val_y, val_predictions_trans, is_val = True)
            # epoch_val_loss += val_loss
            
            # if val_loss - val_drop_threshold >= min_loss:
            #     number_rise_in_amp += 1
            #     if number_rise_in_amp >= val_epoch_threshold:
            #         print("Ending because of overfitting")
            #         return
            # else:
            #     number_rise_in_amp = 0
                
            # if val_loss < min_loss:
            #     min_loss = val_loss
                
            # ### ======= Validation loop ============
            
            if batch%5==0:
                print(loss)
                
        with open(train_loss_file_path,'a') as writer:
            writer.write(str(loss.numpy()) + '\n')
        
        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            model.save(best_model_path)
            
            
        # with open(val_loss_file_path,'a') as writer:
        #     writer.write(str(epoch_val_loss.numpy()) + '\n')
 
        train_generator.on_epoch_end()
        val_generator.on_epoch_end()