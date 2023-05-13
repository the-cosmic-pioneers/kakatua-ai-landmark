#import
import numpy as np 
import pandas as pd 
import PIL 
import tensorflow as tf 
import tensorflow_hub as hub 

# Flask
from flask import Flask, jsonify, request
app = Flask(__name__)


# Load Model & labels
model = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1"
label = "landmarks_classifier_asia_V1_label_map.csv"

@app.route("/")
def home():
    return "App is running"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    

    imageShape = (321,321,3)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model, input_shape = imageShape, output_key="predictions:logits")])

    df = pd.read_csv(label)
    print(df)

    labelsDict = dict(zip(df.id, df.name))  

    image1 = PIL.Image.open(file)
    image1 = image1.resize((321,321))
    print(image1.size)

    image1 = np.array(image1)
    image1 = image1/255.0

    image1 = image1[np.newaxis]

    print(image1.shape)

    result = classifier.predict(image1)
    finalResult = np.argmax(result)

    print ("The prediction is: " + labelsDict[finalResult])
    return jsonify(labelsDict[finalResult])

if __name__ == "__main__":
    app.run()