# preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_and_labels(directory):
    image_paths = []
    labels = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            image_paths.append(img_path)
            labels.append('real' if class_name == 'real' else 'fake')

    return image_paths, np.array(labels)

def split_data_for_cross_validation(image_paths, labels, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_indices = [(train_index, val_index) for train_index, val_index in skf.split(image_paths, labels)]

    return fold_indices


def get_image_generator(image_paths, labels, batch_size, target_size, preprocessing_function, is_train=False):
    if is_train:
        data_gen_args = dict(preprocessing_function=preprocessing_function,
                             horizontal_flip=True,
                             vertical_flip=True, 
                             rotation_range=30,
                             width_shift_range=0.1, 
                             height_shift_range=0.1,
                             zoom_range=0.2) 
    else:
        data_gen_args = dict(preprocessing_function=preprocessing_function)
        
    image_datagen = ImageDataGenerator(**data_gen_args)
    generator = image_datagen.flow_from_dataframe(
                    dataframe=pd.DataFrame({'filename': image_paths, 'class': labels}),
                    class_mode='categorical',
                    y_col='class',
                    x_col='filename',
                    target_size=(target_size, target_size),
                    batch_size=batch_size)
    return generator
