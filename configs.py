"""
All configurations are present in this file
"""
import numpy as np


Configurations = {
    "image_configs": {
            "image_size": (256,256,3),
            "descriptor_length": 128,
            "key_points": np.array([[0.5,0.5],[0,1],[1,0],[0,0],[1,1]])
        },
    
    "paths": {
        "transformed_images_path": "Dataset/img",
        "original_image_path": "test/img/cinema1.jpeg",
        "transformation_matrices_path": "Dataset/trasn",
        "background_images_path": "Dataset/Background",
        "model_path": "model.h5",
        "best_model_path": "best_model.h5",
        "backup_model_path": "model_backup.h5",
        "train_losses_path": "train_loss.txt",
        "val_losses_path": "val_loss.txt",
        "loss_variation_path": "loss_variation.txt",
    },
    
    "training_configs": {
      "lambda": 0.001,
      "mp": 1,
      "mn": 0.2,
      "epochs": 1000,
      "train_batch_size": 40,
      "val_batch_size": 10,
      "val_epoch_threshold": 100,
      "val_drop_threshold": 0,
      "learning_rate": 0.0001,
      "seed" : 12345,  
    },
    
    "inference_configs": {
        "colours": [[0,0,255],[0,255,0],[255,0,0],[255,255,0],[255,0,255],[0,255,255]],
        "inference_thresh": 0.5,
    }
}