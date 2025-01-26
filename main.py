from flask import Flask,render_template,request,jsonify
from keras import models
from keras_preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


model = models.load_model("model\model.h5")
model.load_weights("model\model.weights.h5")

class_labels = {0: "Healthy", 1: "Powdery", 2: "Rust"}  

def preprocess_image(image_path):
  
    img = image.load_img(image_path, target_size=(224, 224))  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route("/", methods=["GET", "POST"])
def landing():
  return render_template("base.html")

@app.route("/learnmore", methods=["GET", "POST"])
def Learnmore():
  return render_template("more.html")

@app.route('/leaflens', methods=['GET', 'POST'])
def leaflens():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return "No file part in the request", 400

        file = request.files['file']
        
        if file.filename == '':
            return "No selected file", 400

        if file:
            
            filename = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            web_filepath = filepath.replace("\\", "/")
            web_filepath = web_filepath.split("static/")[-1]
            web_filepath = f"/static/{web_filepath}"

            print(f"Debug (saved path): {filepath}")       # Local file system path
            print(f"Debug (web path): {web_filepath}")    # Web-accessible URL path (for <img> tag)   

            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            label = class_labels.get(predicted_class, "Unknown")

            return render_template('leaflens.html', prediction=label, confidence=confidence, filepath=web_filepath)

    return render_template('leaflens.html', prediction=None)

if __name__ == "__main__":
  app.run()