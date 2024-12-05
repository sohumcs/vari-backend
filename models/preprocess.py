import os
import cv2
import numpy as np
import pandas as pd

def preprocess_image(image):
    """
    Preprocess image by enhancing contrast, normalizing color, and reducing noise.
    """
    # Convert image to Lab color space (better separation of lightness and color channels)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)  # Enhance only the lightness channel
    enhanced_lab_image = cv2.merge([cl, a, b])
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    # Reduce noise using Gaussian blur for smoothing
    smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    
    return smoothed_image

def calculate_vari(image):
    """
    Calculate the VARI for an image (Green-Red / Green+Red-Blue).
    VARI is sensitive to color balance, so we can adjust for that.
    """
    red = image[:, :, 2].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 0].astype(float)
    
    # Ensure we handle edge cases with a small epsilon to avoid division by zero
    vari = (green - red) / (green + red - blue + 1e-6)  # VARI formula
    return np.clip(vari, -1, 1)  # Clip the values between -1 and 1

def generate_label(vari_mean):
    """
    Generate label based on the mean VARI value.
    """
    if vari_mean >= 0.7:
        return 4  # Very healthy vegetation (green)
    elif vari_mean >= 0.6:
        return 3  # Healthy vegetation
    elif vari_mean >= 0.4:
        return 2  # Semi-healthy vegetation
    elif vari_mean >= 0.2:
        return 1  # Poor vegetation
    else:
        return 0  # Barren or unhealthy vegetation

def split_image_into_regions(image, rows=4, cols=4):
    """
    Split the image into regions of equal size.
    """
    height, width = image.shape[:2]
    region_height = height // rows
    region_width = width // cols
    regions = []

    for i in range(rows):
        for j in range(cols):
            y_start, y_end = i * region_height, (i + 1) * region_height
            x_start, x_end = j * region_width, (j + 1) * region_width
            regions.append((image[y_start:y_end, x_start:x_end], (y_start, y_end, x_start, x_end)))

    return regions

def annotate_image(image, regions, labels):
    """
    Annotate the image with class numbers for each region.
    """
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background for text

    for label, (_, (y_start, y_end, x_start, x_end)) in zip(labels, regions):
        # Calculate text position
        text = str(label)  # Only the class number
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x_start + (x_end - x_start - text_size[0]) // 2
        text_y = y_start + (y_end - y_start + text_size[1]) // 2

        # Draw background rectangle for better visibility
        cv2.rectangle(
            annotated_image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            bg_color,
            thickness=-1,
        )
        # Add text annotation
        cv2.putText(annotated_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return annotated_image

def enhance_image(image):
    """
    This function aims to enhance the image based on global illumination or low-contrast regions.
    """
    # Apply a simple brightness/contrast enhancement
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    l = cv2.equalizeHist(l)  # Equalize histogram of the L (lightness) channel
    enhanced_image = cv2.merge([l, a, b])
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Directories
input_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/images"
output_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/output_images"
vari_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/vari_values"
label_folder = "C:/Users/Sohum Srivastava/Desktop/vari-app/project/backend/labels"


# Process each image
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load and preprocess image
        image = cv2.imread(filepath)
        enhanced_image = enhance_image(image)  # Brightness/contrast adjustment
        preprocessed_image = preprocess_image(enhanced_image)

        # Calculate VARI for vegetation detection
        vari = calculate_vari(preprocessed_image)
        
        # Split image into regions
        regions = split_image_into_regions(image, 4, 4)

        # Calculate mean VARI for each region
        vari_values = [np.mean(vari[y_start:y_end, x_start:x_end]) for _, (y_start, y_end, x_start, x_end) in regions]
        labels = [generate_label(value) for value in vari_values]

        # Save VARI values to CSV
        pd.DataFrame([vari_values], columns=[f"region_{i+1}" for i in range(16)]).to_csv(
            os.path.join(vari_folder, f"{os.path.splitext(filename)[0]}.csv"), index=False
        )

        # Save labels to TXT
        with open(os.path.join(label_folder, f"{os.path.splitext(filename)[0]}.txt"), "w") as f:
            f.write(','.join(map(str, labels)))

        # Annotate the image with class labels
        annotated_image = annotate_image(image, regions, labels)

        # Save the annotated image
        cv2.imwrite(os.path.join(output_folder, filename), annotated_image)

        print(f"Processed and saved: {filename}")
