#Import
import numpy as np 
import pandas as pd 
import PIL 
import tensorflow as tf 
import tensorflow_hub as hub 
from flask import Flask, jsonify, request

#Flask
app = Flask(__name__)


#Load Models & labels
model = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1"
label = "./landmarks_classifier_asia_V1_label_map.csv"


#Check for app running
@app.route("/")
def home():
    return "App is running"


#Process Predict Landmark
@app.route("/upload", methods=["POST"])
def upload():

    #Check file
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    

    imageShape = (321,321,3)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model, input_shape = imageShape, output_key="predictions:logits")])

    df = pd.read_csv(label)

    labelsDict = dict(zip(df.id, df.name))  

    image1 = PIL.Image.open(file)
    image1 = image1.resize((321,321))

    image1 = np.array(image1)
    image1 = image1/255.0

    image1 = image1[np.newaxis]

    result = classifier.predict(image1)
    finalResult = np.argmax(result)

    return jsonify({"landmark": labelsDict[finalResult]})

if __name__ == "__main__":
    app.run()