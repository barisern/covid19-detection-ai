from flask import Flask, render_template, url_for, request, redirect
from werkzeug.utils import secure_filename
import os

import tensorflow as tf
import cv2

model = tf.keras.models.load_model('./static/model/covid-ct-cnn-64x3-512-last-1614079489.h5')
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

@app.route('/')
def home():
    return render_template('index.html', page='home')

@app.route('/about')
def about():
    return render_template('about.html', page='about')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'ctimage' not in request.files:
            return redirect(request.url)
            
        file = request.files['ctimage']

        if file.filename == '':
            return redirect(request.url)

        if file and check_file_ext(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            res = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template('predict.html', image=filename, result=res)
    else:
        ex_file = request.args.get('example', default='ct1.png', type=str)
        if ex_file == 'ct1.png' or ex_file == 'ct2.png' or ex_file == 'ct3.png':
            res = predict(os.path.join(app.config['UPLOAD_FOLDER'], ex_file)) 
            return render_template('predict.html', image=ex_file, result=res)
            
    return render_template('index.html', page='home')



# Other functions
def check_file_ext(filename):
    valid_ext = set(['png', 'jpg', 'jpeg', 'jfif'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in valid_ext


def prepare(path):
  IMG_SIZE = 200
  img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
  return resized_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predict(filename):
    CATEGORIES = ['COVID', 'non-COVID']

    prediction = model.predict([prepare(filename)])

    return CATEGORIES[int(prediction[0][0])]
