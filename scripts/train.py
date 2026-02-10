import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Setup - Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
DATASET_PATH = "../data"  # Place your image folders here

def build_model(num_classes):
    # Data Augmentation Layer - Shows you know how to handle small datasets
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    # Base Model: MobileNetV2 (Lightweight for Edge/Production)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the weights

    # Building the final architecture
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) # Regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def main():
    # Load Dataset from local folders
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    num_classes = len(train_ds.class_names)
    model = build_model(num_classes)

    # Compile with Adam Optimizer
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save the model
    os.makedirs('../models', exist_ok=True)
    model.save('../models/plant_model_v1.keras')
    print("Model saved to ../models/plant_model_v1.keras")

if __name__ == "__main__":
    main()
