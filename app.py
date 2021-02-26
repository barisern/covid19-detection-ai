from flask import Flask, render_template, url_for, request, redirect
from werkzeug.utils import secure_filename

import os
import uuid
import random

import cv2
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

model = tf.keras.models.load_model('./static/model/covid-ct-cnn-64x3-512-last-1614079489.h5')
examples = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'examples'))

@app.route('/')
def home():
    return render_template('index.html', page='home', examples=random.sample(examples, 3))

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
            file_ext = os.path.splitext(filename)[1]

            filename = str(uuid.uuid4()) + file_ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)

            res = predict(filepath)

            return render_template('predict.html', image=filename, result=res)
    else:
        ex_file = request.args.get('example', default=examples[0], type=str)

        if ex_file in examples:
            res = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'examples', ex_file)) 
            return render_template('predict.html', image=os.path.join('examples', ex_file), result=res)
         
    return redirect('/')

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
