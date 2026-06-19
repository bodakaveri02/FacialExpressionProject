import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
TRAIN_DIR = 'train'
TEST_DIR = 'test'
MODEL_PATH = 'emotion_model.h5'

def train_model():
    print("Checking dataset directories...")
    
    # Check if folders exist
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: '{TRAIN_DIR}' folder not found!")
        print("Please ensure the 'train' folder is inside your project folder.")
        return

    # Data generators to load images automatically from folders
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training images...")
    train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    print("Loading test images...")
    validation_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    # Build the CNN Model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    print("Starting training...")
    print("This may take time. Please wait...")
    
    # Train the model
    model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // 64,
            epochs=10, 
            validation_data=validation_generator,
            validation_steps=validation_generator.n // 64
    )

    # Save the model
    model.save(MODEL_PATH)
    print(f"Training Complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()