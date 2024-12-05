from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("C:/Users/Sohum Srivastava/Desktop/vari-backend/backend/trained_model.h5")  # Pre-trained model

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    file_path = os.path.join('images', file.filename)
    file.save(file_path)

    # Preprocess the image
    image = cv2.imread(file_path)
    processed_image = preprocess_image(image)  # Assuming you have this function

    # Classify and annotate
    class_labels = classify_image_parts(image, model)  # Assuming you have this function
    annotated_image = overlay_class_labels(image, class_labels)  # Assuming you have this function
    output_path = os.path.join('output_images', file.filename)
    cv2.imwrite(output_path, annotated_image)

    return jsonify({"output_path": output_path, "classes": class_labels}), 200

# This is a Vercel-specific function handler
def handler(request):
    return app(request)
