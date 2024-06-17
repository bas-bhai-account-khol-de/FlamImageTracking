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
backup_model_path = Configurations["paths"]["backup_model_path"]


class ChannelNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(ChannelNormalization, self).__init__()

    def call(self, inputs):
        norm = tf.norm(inputs, axis=-1, keepdims=True)
        norm = tf.where(norm != 0, norm, 1)
        normalized_inputs = inputs / norm
        return normalized_inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        super(ChannelNormalization, self).build(input_shape)
        
    def get_config(self):
        return super(ChannelNormalization, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        instance = cls()
        return instance
    
class ResizingLayer(tf.keras.layers.Layer):
    def __init__(self, outputsize, **kwargs):
        super(ResizingLayer, self).__init__(**kwargs)
        self.outputsize = outputsize
    
    def call(self, feature_map):
        return tf.image.resize(feature_map, self.outputsize, method='bicubic')
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.outputsize[0], self.outputsize[1], input_shape[3])
    
    def build(self, input_shape):
        super(ResizingLayer, self).build(input_shape)
        
    def get_config(self):
        config = super(ResizingLayer, self).get_config()
        config.update({
            'outputsize': self.outputsize,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        instance = cls(config.pop("outputsize"))
        return instance
    
class DroppingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DroppingLayer, self).__init__(**kwargs)
    
    def call(self, feature_map):
        return feature_map[:,:,:,:64]
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 64)
    
    def build(self, input_shape):
        super(DroppingLayer, self).build(input_shape)
        
    def get_config(self):
        config = super(DroppingLayer, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        instance = cls()
        return instance
    
class DeformableConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1), activation = "linear",**kwargs):
        super(DeformableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        
        # Define the convolution for generating offsets
        self.offset_conv = tf.keras.layers.Conv2D(filters=2,# * kernel_size[0] * kernel_size[1],
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           dilation_rate=dilation_rate,
                                           kernel_initializer='zeros',
                                           bias_initializer='zeros')
        
        # Define the convolution for deformable convolution
        self.deform_conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=(1, 1),
                                           padding='same',
                                           dilation_rate=(1, 1))

    def call(self, inputs):
        offsets = self.offset_conv(inputs)
        outputs = self.deform_conv(self._apply_offsets(inputs, offsets))
        return outputs
    
    def _apply_offsets(self, inputs, offsets):
        batch_size, height, width, channels = tf.shape(inputs)
        offset_h, offset_w = tf.split(offsets, 2, axis=-1)
        
        # Generate a meshgrid to apply the offsets
        y, x = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
        y = tf.cast(y, dtype=offsets.dtype)
        x = tf.cast(x, dtype=offsets.dtype)
        y = tf.reshape(y, (1,y.shape[0], y.shape[1], 1))
        y = np.tile(y, (offset_h.shape[0],1,1,offset_h.shape[3]))
        
        x = tf.reshape(x, (1,x.shape[0], x.shape[1], 1))
        x = np.tile(x, (offset_w.shape[0],1,1,offset_w.shape[3]))
        
        y = y + offset_h
        x = x + offset_w
        
        y = tf.clip_by_value(y, 0, tf.cast(height - 1, dtype=y.dtype))
        x = tf.clip_by_value(x, 0, tf.cast(width - 1, dtype=x.dtype))
        
        batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1, 1, 1))
        batch_indices = np.tile(batch_indices, (1, height, width, offset_h.shape[3]))
    
        indices = tf.stack([batch_indices, y, x], axis=-1)
        indices = tf.reshape(indices, (-1, 3))
        indices = tf.cast(indices, dtype=tf.int32)
        
        output = tf.gather_nd(inputs, indices)
        output = tf.reshape(output, (batch_size, height, width, channels))
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)
    
    def build(self, input_shape):
        super(DeformableConv2D, self).build(input_shape)
    
    def get_config(self):
        config = super(DeformableConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'dilation_rate': self.dilation_rate,
            'offset_config': tf.keras.layers.serialize(self.offset_conv),
            'deform_config': tf.keras.layers.serialize(self.deform_conv)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        layer1 = tf.keras.layers.deserialize(config.pop('offset_config'))
        layer2 = tf.keras.layers.deserialize(config.pop('deform_config'))
        
        instance = cls(config.pop('filters'),
                       config.pop('kernel_size'),
                       config.pop('strides'),
                       config.pop('padding'),
                       config.pop('dilation_rate'),
                       config.pop('activation'))
        
        instance.offset_conv = layer1
        instance.deform_conv = layer2
        return instance
    
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

def get_corresponding_points(key_points_orig, descriptors_orig, descriptors_trans):
    orig_descs = descriptors_orig[0, key_points_orig[:, 0], key_points_orig[:, 1], :]
    trans_descs = descriptors_trans[0].reshape(-1, descriptors_trans.shape[-1])
    cosine_similarities = np.dot(orig_descs, trans_descs.T)
    max_indices = np.argmax(cosine_similarities, axis=1)
    key_points_trans = np.column_stack(np.unravel_index(max_indices, (descriptors_trans.shape[1], descriptors_trans.shape[2])))
    return key_points_trans

def images_to_patches(images):
    n, h, w = images.shape
    patches = images.reshape(n, h//8, 8, w//8, 8).transpose(0, 1, 3, 2, 4).reshape(n, h//8, w//8, 64)
    return patches

def patches_to_images(patches):
    n, p1, p2, c = patches.shape
    images = patches.reshape(n, p1, p2, 8, 8)
    images = tf.transpose(images,(0, 1, 3, 2, 4)).reshape(n, image_size[0], image_size[0])
    return images

def detection_loss(probability_true, probability_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(probability_true, probability_pred)
    return tf.keras.ops.mean(loss)


def descriptor_loss(orig_descriptor, trans_descriptor):
    a, b, c, d = orig_descriptor.shape
    
    j_coords, k_coords = np.meshgrid(np.arange(b), np.arange(c), indexing='ij')
    l_coords, m_coords = np.meshgrid(np.arange(b), np.arange(c), indexing='ij')
    
    dist_sq = (j_coords[:, :, None, None] - l_coords[None, None, :, :])**2 + (k_coords[:, :, None, None] - m_coords[None, None, :, :])**2
    s_mask = dist_sq < 64
    
    total_sum = 0.0
    for i in range(a):
        orig_flat = orig_descriptor[i].reshape(b * c, d)
        trans_flat = trans_descriptor[i].reshape(b * c, d)
        dot_products = np.dot(orig_flat, trans_flat.T)
        dot_products = dot_products.reshape(b, c, b, c)
        s_term = s_mask * np.maximum(0, mp - dot_products)
        non_s_term = (~s_mask) * np.maximum(0, dot_products - mn)
        total_sum += np.sum(s_term + non_s_term)
    return total_sum / ((b * c) ** 2)

def create_65_layers(map):
    map = images_to_patches(map)
    layer_65 = np.ones((*map.shape[:-1], 1), dtype=int)
    all_zeros = np.all(map == 0, axis= -1)
    layer_65[:,:,:,0] = all_zeros
    map = np.concatenate([map, layer_65], axis = -1)
    return map
 
def custom_loss(y_true, predictions_orig, predictions_trans, is_val = False):
    probability_true, key_points_true = y_true[:,:,0], y_true[:,:,1:] 
    orig_detect, orig_descriptor = predictions_orig
    trans_detect, trans_descriptor = predictions_trans
    
    ## Transformed points into maps
    trans_true_kps_map = np.zeros((probability_true.shape[0],*image_size))
    for i in range(probability_true.shape[0]):
        for j in range(probability_true.shape[1]):
            if probability_true[i][j]:
                x = int(image_size[0]*key_points_true[i][j][0])
                x = 0 if x == 0 else x - 1
                y = int(image_size[1]*key_points_true[i][j][1])
                y = 0 if y == 0 else y - 1
                trans_true_kps_map[i][x][y] = 1
    trans_true_kps_map = create_65_layers(trans_true_kps_map)
    
    ## Original points into maps
    orig_true_kps_map = np.zeros((probability_true.shape[0],*image_size))
    for i in range(probability_true.shape[0]):
        for j in range(keypoints.shape[0]):
            x = int(image_size[0]*keypoints[j][0])
            x = 0 if x == 0 else x - 1
            y = int(image_size[1]*keypoints[j][1])
            y = 0 if y == 0 else y - 1
            orig_true_kps_map[i][x][y] = 1
    orig_true_kps_map = create_65_layers(orig_true_kps_map)
    
    orig_detection_loss = detection_loss(orig_true_kps_map, orig_detect)
    trans_detection_loss = detection_loss(trans_true_kps_map, trans_detect)
    combined_descriptor_loss = descriptor_loss(orig_descriptor, trans_descriptor)
    
    if not is_val:
        with open(loss_variation_file_path,'a') as writer:
            writer.write(f"Orig_detection_loss: {orig_detection_loss}, trans_detection_loss: {trans_detection_loss}, combined_descriptor_loss: {combined_descriptor_loss} \n")
            
    return orig_detection_loss + trans_detection_loss #+ (weight * combined_descriptor_loss)
                    
def train(train_generator:CustomDataGenerator, val_generator:CustomDataGenerator, model, epochs, optimizer):
    with open(train_loss_file_path,'w') as writer:
        writer.write('')
    with open(val_loss_file_path,'w') as writer:
        writer.write('')
    with open(loss_variation_file_path,'w') as writer:
            writer.write('')
    
    min_loss = np.inf
    number_rise_in_amp = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch} starting...")
        for batch in range(train_generator.__len__()):
            x, y = train_generator.__getitem__(batch)
            with tf.GradientTape() as tape:
                predictions_orig = model(x[0])
                predictions_trans = model(x[1])
                loss = custom_loss(y, predictions_orig, predictions_trans)  
                with open(train_loss_file_path,'a') as writer:
                    writer.write(str(loss.numpy()) + '\n')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            try:
                model.save(model_path)
                model.save(backup_model_path)
            except Exception as e:
                print("========= Note ================")
                print("cant save last model")
                print(e)
            
            
            ### ======= Validation loop ============
            val_x, val_y = val_generator.__getitem__(np.random.randint(0,val_generator.__len__()))
            val_predictions_orig = model(val_x[0])
            val_predictions_trans = model(val_x[1])
            val_loss = custom_loss(val_y, val_predictions_orig, val_predictions_trans, is_val = True)
            with open(val_loss_file_path,'a') as writer:
                    writer.write(str(val_loss.numpy()) + '\n')
            if val_loss - val_drop_threshold >= min_loss:
                number_rise_in_amp += 1
                if number_rise_in_amp >= val_epoch_threshold:
                    print("Ending because of overfitting")
                    return
            else:
                number_rise_in_amp = 0
                
            if val_loss < min_loss:
                min_loss = val_loss
                
            ### ======= Validation loop ============
            
            if batch%5==0:
                print(loss)
                
        train_generator.on_epoch_end()
        val_generator.on_epoch_end()