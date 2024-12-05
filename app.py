from flask import Flask, request, jsonify
from models.preprocess import preprocess_image, enhance_image
from models.train_model import build_model, train_model
from models.test_model import classify_image_parts, overlay_class_labels
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("trained_model.h5")  # Pre-trained model

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    file_path = os.path.join('images', file.filename)
    file.save(file_path)

    # Preprocess the image
    image = cv2.imread(file_path)
    processed_image = preprocess_image(image)

    # Classify and annotate
    class_labels = classify_image_parts(image, model)
    annotated_image = overlay_class_labels(image, class_labels)
    output_path = os.path.join('output_images', file.filename)
    cv2.imwrite(output_path, annotated_image)

    return jsonify({"output_path": output_path, "classes": class_labels}), 200

if __name__ == '__main__':
    app.run(debug=True)
