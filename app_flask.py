import os
from flask import Flask, request
from werkzeug.utils import secure_filename
import sys
from os.path import abspath, dirname
from evaluation.evaluate import evaluate_image
from functions import make_dir_if_not_exists, remove_file_if_exists

sys.path.insert(0, dirname(dirname(abspath(__file__))))

app = Flask(__name__)
TOKEN = "image"
IMAGE_FOLDER = 'STATIC/uploads'
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # byte (1MB)

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


make_dir_if_not_exists(IMAGE_FOLDER)

@app.route("/")
def hello():
    return "Working!"

@app.route('/image_uploader', methods=['GET', 'POST'])
def upload_file():

    if request.headers.get('token',str) == TOKEN:
        if request.method == 'POST':
            f = request.files['file']
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['IMAGE_FOLDER'], filename)
                f.save(path)
                image_classifier = evaluate_image(path)
                remove_file_if_exists(path)
            else:
                return "File Format is not Compatible"
            return image_classifier
    else:
        return "Could Not Verify The Token"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')