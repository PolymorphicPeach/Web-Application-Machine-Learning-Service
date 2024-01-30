import flask
from flask import Flask, request, jsonify
import requests
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO
import numpy as np
import json

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
aerial_image_labels = ["BareLand", "Beach", "DenseResidential", "Desert",
                       "Forest", "Mountain", "Parking", "SparseResidential"]

app = Flask(__name__)

aerial_model = tf.keras.models.load_model("static/aerial_classifier.keras")
aerial_model.summary()

@app.route("/")
def version():
    tensorflow_version = tf.__version__
    return f"<p> Tensorflow version: {tensorflow_version} </p>"

@app.route("/aerial", methods=['POST'])
def aerial_classification():
    response_data = ""
    request_body = request.get_data(as_text=True)
    url = request_body.strip()

    # Image comes in "P" (Palette format), must convert to RGB
    googleResponse = requests.get(url)
    if googleResponse.status_code == 200:
        image = Image.open(BytesIO(googleResponse.content))
        image = image.convert("RGB")
        # Scale pixels
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        prediction = aerial_model.predict(image_array)
        rounded_probabilities = [round(float(probability), 3) for probability in prediction[0]]

        response_data = {
            class_label: float(probability)
            for class_label, probability in zip (aerial_image_labels, rounded_probabilities)
        }

    return json.dumps(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
