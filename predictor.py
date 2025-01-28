from keras import Model
import keras
from keras_preprocessing import image
import numpy as np

model = keras.models.load_model('E:\General\Programming\Python\ML from Scratch\SIHModel\model.h5')
model.load_weights('E:\General\Programming\Python\ML from Scratch\SIHModel\model.weights.h5')

image_path = 'E:\General\Programming\Python\ML from Scratch\SIHModel\8ddaa5a5caa5caa8.jpg'  
img = image.load_img(image_path, target_size=(224, 224))  

img_array = image.img_to_array(img)

img_array = img_array / 255.0  

img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])  
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
