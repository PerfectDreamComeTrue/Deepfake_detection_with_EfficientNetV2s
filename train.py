# train.py
import keras
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from preprocessing import load_images_and_labels, get_image_generator, split_data_for_cross_validation
from tensorflow.keras import mixed_precision

# GPU Settings
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Constants
IMG_SIZE = 384
NUM_CLASSES = 2
SEED = 1
N_FOLDS = 5 #defualt
EPOCHS = 80
BATCH_SIZE  = 123
dropout_rate = 0.5

# Paths
dataset_dir = "Deepfake_detection_with_EfficientNetV2s/dataset/train"
model_dir = '/Deepfake_detection_with_EfficientNetV2s/Checkpoint/Models'
history_dir = '/Deepfake_detection_with_EfficientNetV2s/Checkpoint/History'

# Set seed for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load the data
image_paths, labels = load_images_and_labels(dataset_dir)

# Create stratified k-folds
fold_indices = split_data_for_cross_validation(image_paths, labels, n_folds=N_FOLDS)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


# Train and validate the model for each fold
for fold, (train_index, val_index) in enumerate(fold_indices):
    print(f"Training on fold {fold+1}")
  
    # Create the model
    base_model = EfficientNetV2S(weights= 'imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Prepare the data generators
    train_images = np.array(image_paths)[train_index]
    train_labels = labels[train_index]
    val_images = np.array(image_paths)[val_index]
    val_labels = labels[val_index]

    train_gen = get_image_generator(train_images, train_labels, BATCH_SIZE , IMG_SIZE, preprocess_input,is_train=True)
    val_gen = get_image_generator(val_images, val_labels, BATCH_SIZE , IMG_SIZE, preprocess_input, is_train=False)

    # Compute class weights
    weights = class_weight.compute_sample_weight('balanced', train_labels)
    class_weights = dict(enumerate(weights))

    # Prepare callbacks
    checkpoint = ModelCheckpoint(f"{model_dir}/model_fold_{fold+1}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, min_delta=1e-3, mode='min')
    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # Compile and train the model
    model.compile(optimizer=Adam(learning_rate =1e-2), loss='categorical_crossentropy', metrics=['accuracy', F1Score()])

    history = model.fit(
        train_gen,
        epochs=EPOCHS,  
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1)


    # Save the history
    np.save(f"{history_dir}/history_fold_{fold+1}.npy", history.history)

    # Save the validation data for performance analysis
    np.save(f"{history_dir}/val_images_fold_{fold+1}.npy", val_images)
    np.save(f"{history_dir}/val_labels_fold_{fold+1}.npy", val_labels)
