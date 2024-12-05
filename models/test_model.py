import tensorflow as tf
import cv2
import os
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("trained_model.h5")

# Set class colors for visualization (Optional for debugging)
class_colors = {
    0: (0, 0, 0),       # Black
    1: (0, 0, 255),     # Red
    2: (0, 255, 255),   # Yellow
    3: (0, 255, 0),     # Green
    4: (255, 0, 0)      # Blue
}

def preprocess_image(image_path):
    """
    Load and preprocess the image.
    Resizes to 128x128 and normalizes pixel values.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    return np.expand_dims(image_resized, axis=0), image

def split_image_into_parts(image, num_parts=16):
    """
    Split the image into equal parts (4x4 grid in this case).
    Assumes image height and width can be evenly divided by num_parts.
    """
    height, width, _ = image.shape
    part_height = height // 4
    part_width = width // 4
    parts = []
    for i in range(4):
        for j in range(4):
            part = image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            parts.append(part)
    return parts

def overlay_class_labels(image, class_labels):
    """
    Overlay the class labels onto the image with the actual class number for each region.
    """
    height, width, _ = image.shape
    part_height = height // 4
    part_width = width // 4
    idx = 0
    for i in range(4):
        for j in range(4):
            y_start = i * part_height
            y_end = (i + 1) * part_height
            x_start = j * part_width
            x_end = (j + 1) * part_width
            color = class_colors[class_labels[idx]]
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, -1)  # Overlay color for the region
            cv2.putText(image, str(class_labels[idx]), (x_start + 10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            idx += 1
    return image

def classify_image_parts(image):
    """
    Divide the image into parts and classify each part.
    """
    parts = split_image_into_parts(image)
    class_labels = []
    
    for part in parts:
        part_resized = cv2.resize(part, (128, 128))  # Ensure consistency in image size
        part_resized = part_resized / 255.0
        part_input = np.expand_dims(part_resized, axis=0)
        prediction = model.predict(part_input)
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        class_labels.append(predicted_class)
    
    return class_labels

# Path to the folder containing test images
test_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/images"
output_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/test_out"


# Loop through images in the test folder
for filename in os.listdir(test_folder):
    filepath = os.path.join(test_folder, filename)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            image_input, original_image = preprocess_image(filepath)
            predictions = model.predict(image_input)

            # Visualize the classification results
            parts_class = classify_image_parts(original_image)
            print(f"Classified parts for {filename}: {parts_class}")

            # Overlay class labels and actual class values onto the image
            result_image = overlay_class_labels(original_image.copy(), parts_class)

            # Save the resulting image with annotations
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, result_image)
            print(f"Saved classified image to {output_image_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
