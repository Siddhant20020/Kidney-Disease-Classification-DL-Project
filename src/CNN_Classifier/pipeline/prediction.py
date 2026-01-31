
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from cnnClassifier.utils.common import decodeImage
from tensorflow.keras.applications.resnet50 import preprocess_input


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app = Flask(__name__)
CORS(app)


CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"] 

class PredictionPipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.target_size = (224, 224)

    
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from cnnClassifier.utils.common import decodeImage
from tensorflow.keras.applications.resnet50 import preprocess_input


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app = Flask(__name__)
CORS(app)


CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"] 

class PredictionPipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.target_size = (224, 224)

    def predict(self, image_path):
        from tensorflow.keras.preprocessing import image
        # Load and preprocess
        img = image.load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = self.model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        return {
            "class": CLASS_NAMES[class_idx],
            "confidence": float(np.max(preds))
        }

# Client app
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline("artifacts/training/model.h5")

# Flask routes
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")  
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict(clApp.filename)
    return jsonify(result)

# Run app
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080, debug=True)


# Client app
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline("artifacts/training/model.h5")

# Flask routes
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")  
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict(clApp.filename)
    return jsonify(result)

# Run app
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080, debug=True)
