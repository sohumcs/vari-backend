import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Reshape
from tensorflow.keras.utils import to_categorical
                                
# Set paths
image_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/images"
label_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/labels"
num_classes = 5  # Number of classes (adjust this as per your dataset)
num_parts = 16   # 4x4 grid = 16 regions

def load_data(image_folder, label_folder):
    images, labels = [], []

    for file_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, file_name)
        label_path = os.path.join(label_folder, file_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        images.append(image)

        try:
            with open(label_path, 'r') as f:
                label = list(map(int, f.read().strip().split(',')))
            if len(label) != num_parts:
                print(f"Skipping {file_name}: Label count {len(label)} != {num_parts}")
                continue
            labels.append(to_categorical(label, num_classes=num_classes))
        except FileNotFoundError:
            print(f"Label file not found for {file_name}, skipping...")

    return np.array(images), np.array(labels)

def build_model():
    model = Sequential([
        InputLayer(input_shape=(128, 128, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_parts * num_classes, activation='softmax'),
        Reshape((num_parts, num_classes))  # Reshape to (16, num_classes)
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, images, labels):
    labels = labels.reshape(-1, num_parts, num_classes)  # Ensure correct shape
    model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save("trained_model.h5")
    print("Model training complete. Model saved to 'trained_model.h5'.")

if __name__ == "__main__":
    images, labels = load_data(image_folder, label_folder)
    if images.size == 0 or labels.size == 0:
        print("No valid data found. Exiting.")
    else:
        model = build_model()
        train_model(model, images, labels)
