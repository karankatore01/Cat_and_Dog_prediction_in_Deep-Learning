from flask import Flask, render_template, request,redirect
import os
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model(r'D:\Projects\Deep learning\Cat dog prediction\cnn_own_vgg16.h5')

def predict_cat_or_dog(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224)) / 255
    yp = model.predict_on_batch(img.reshape(-1, 224, 224, 3)).argmax()
    return 'CAT' if yp == 0 else 'DOG'


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_path = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)
            prediction = predict_cat_or_dog(file_path)

    return render_template('index.html', prediction=prediction, file_path=file_path)

@app.route('/')
def display(filename):
    print('Display img'+filename)
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
