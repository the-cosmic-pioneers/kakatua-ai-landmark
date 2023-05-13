#import
import numpy as np 
import pandas as pd 
import PIL 
import tensorflow as tf 
import tensorflow_hub as hub 

model = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1"
label = "landmarks_classifier_asia_V1_label_map.csv"

imageShape = (321,321,3)
classifier = tf.keras.Sequential(
    [hub.KerasLayer(model, input_shape = imageShape, output_key="predictions:logits")])

df = pd.read_csv(label)
print(df)

labelsDict = dict(zip(df.id, df.name))

image1 = PIL.Image.open("monas.jpg")
image1 = image1.resize((321,321))
print(image1.size)

image1 = np.array(image1)
image1 = image1/255.0

image1 = image1[np.newaxis]

print(image1.shape)

result = classifier.predict(image1)
finalResult = np.argmax(result)

print ("The prediction is: " + labelsDict[finalResult])