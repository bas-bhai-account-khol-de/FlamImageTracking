import tensorflow as tf
from PointLocator import load_and_display_trsnformed_image
import numpy as np
import cv2
import pickle
import math
import os
import warnings
from configs import Configurations

# tf.experimental.numpy.experimental_enable_numpy_behavior()
warnings.simplefilter('ignore', category=FutureWarning)

image_size = Configurations["image_configs"]["image_size"][:2]
keypoints = Configurations["image_configs"]["key_points"]

val_epoch_threshold = Configurations["training_configs"]["val_epoch_threshold"]
val_drop_threshold = Configurations["training_configs"]["val_drop_threshold"]
weight = Configurations["training_configs"]["lambda"]
mp = Configurations["training_configs"]["mp"]
mn = Configurations["training_configs"]["mn"]

synthetic_shapes_path = Configurations["paths"]["synthetic_shapes"]
pseudo_keypoints_path = Configurations["paths"]["pseudo_keypoints"]
original_images_path = Configurations["paths"]["original_image_path"]
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
        
        self.batchsize = batch_size
        self.synthetic_shapes = []
        self.keypoints = []
        for file in os.listdir(synthetic_shapes_path):
            if file.endswith("jpeg"):
                self.synthetic_shapes.append(os.path.join(synthetic_shapes_path, file))
                
        for file in os.listdir(pseudo_keypoints_path):
            if file.endswith(".pkl"):
                with open(os.path.join(pseudo_keypoints_path,file), 'rb') as f:
                    kp = np.array(pickle.load(f))
                    self.keypoints.append(kp)
        
        self.indices = np.arange(0,len(self.keypoints))
                      
    def __len__(self):
        return math.ceil(len(self.keypoints)/self.batchsize)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def preprocess_keypoints(self, keypoints):
        gt = np.zeros((1, *image_size))
        if (keypoints.shape[0] > 0):
            if (keypoints.shape[1] == 2):
                for kp in keypoints:
                    gt[0][kp[1],kp[0]] = 1
                    
        gt = create_65_layers(gt)
        return gt[0]
            
    def __getitem__(self, idx):
        low = idx * self.batchsize
        high = min(low + self.batchsize, len(self.keypoints))
        
        needed_indices = self.indices[low:high]
        
        batch_shapes = []
        for path in [self.synthetic_shapes[i] for i in needed_indices]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis = -1)
            batch_shapes.append(img)
        batch_keypoints = []
        for keypoints in [self.keypoints[i] for i in needed_indices]:
            batch_keypoints.append(self.preprocess_keypoints(keypoints))
            
            
        return np.array(batch_shapes), np.array(batch_keypoints)

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
    # loss = tf.keras.losses.MeanSquaredError()
    loss = tf.keras.losses.CategoricalCrossentropy()
    return loss(probability_true, probability_pred)

def descriptor_loss(orig_descriptor, trans_descriptor, homographies):
    a, b, c, d = orig_descriptor.shape
    loss = 0.0

    valid = tf.Variable(tf.zeros((a,b,c,b,c)))
    dot_products = tf.Variable(tf.zeros((a,b,c,b,c)))
    
    for image in range(a):
        with open(homographies[image], 'rb') as f:
            homography = pickle.load(f)
        for i in range(b):
            for j in range(c):
                for k in range(b):
                    for l in range(c):
                        changed_points = tf.matmul(homography, tf.convert_to_tensor([[j], [i], [0], [1]], dtype=tf.float32))
                        changed_points = tf.convert_to_tensor([changed_points[1] / changed_points[2], changed_points[0] / changed_points[2]])
                        
                        # Conditionally update valid tensor using tf.where
                        valid = tf.cast(tf.abs(changed_points[0] - tf.cast(k, tf.float32)) + tf.abs(changed_points[1] - tf.cast(l, tf.float32)) <= 8, dtype=float)
                        
        dot_products = tf.einsum('aijd,akld->aijkl', orig_descriptor, trans_descriptor)
    
    loss = (valid * tf.maximum(0, mp - dot_products)) + ((1-valid) * tf.maximum(0, dot_products - mn))
    
    return tf.reduce_mean(loss)
   
def create_65_layers(map):
    map = images_to_patches(map)
    # ones_mask = tf.equal(map, 1)
    
    # # Cumulative sum along the last dimension to identify the first 1
    # cumsum_ones = tf.cumsum(tf.cast(ones_mask, tf.int32), axis=-1)
    
    # # Create a mask to keep only the first 1 in each last dimension slice
    # first_one_mask = tf.equal(cumsum_ones, 1)
    
    # # Apply the mask to the original array to keep only the first 1s
    # processed_arr = tf.where(first_one_mask, tf.ones_like(map), tf.zeros_like(map))
    
    layer_65 = np.ones((*map.shape[:-1], 1), dtype=int)
    all_zeros = np.all(map == 0, axis= -1)
    layer_65[:,:,:,0] = all_zeros
    map = np.concatenate([map, layer_65], axis = -1)
    return map
 
def custom_loss(y_true, predictions_orig, predictions_trans, homography_matrix, mode, is_val = False):
    
    if mode == Configurations["training_configs"]["training_mode"][0]:
        return detection_loss(y_true, predictions_orig[0])
    probability_true, key_points_true = y_true[:,:,0], y_true[:,:,1:] 
    orig_detect, orig_intermediate_descriptor, orig_descriptor = predictions_orig
    trans_detect, trans_intermediate_descriptor, trans_descriptor = predictions_trans
    
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
            orig_true_kps_map[i][y][x] = 1
    orig_true_kps_map = create_65_layers(orig_true_kps_map)
    
    
    orig_detection_loss = detection_loss(orig_true_kps_map, orig_detect)
    trans_detection_loss = detection_loss(trans_true_kps_map, trans_detect)
    combined_descriptor_loss = descriptor_loss(orig_intermediate_descriptor, trans_intermediate_descriptor, homography_matrix)
    
    if not is_val:
        with open(loss_variation_file_path,'a') as writer:
            writer.write(f"Orig_detection_loss: {orig_detection_loss}, trans_detection_loss: {trans_detection_loss}, combined_descriptor_loss: {combined_descriptor_loss} \n")
            
    return combined_descriptor_loss #orig_detection_loss + trans_detection_loss # + (weight * combined_descriptor_loss)
                    
def train(train_generator:CustomDataGenerator, val_generator:CustomDataGenerator, model, epochs, optimizer, mode):
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
                predictions_orig = model(x)
                loss = custom_loss(y, predictions_orig, None, None, mode)
                
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
            
            if batch%5==0:
                print(loss)
                
        train_generator.on_epoch_end()
        val_generator.on_epoch_end()