import os
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File
import shutil
from configs.config_ml_model import ROOT_DIR
from evaluation.evaluate import evaluate_image
from functions import make_dir_if_not_exists, remove_file_if_exists

app = FastAPI()

IMAGE_FOLDER = 'STATIC/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.get('/')
async def home():
    return{"message" : "Working"}


@app.post('/image')
async def image(image: UploadFile = File(...)):
    make_dir_if_not_exists(IMAGE_FOLDER)
    # return{"filename" : image.filename}
    if image and allowed_file(image.filename):
        path = os.path.join(IMAGE_FOLDER, image.filename)
        with open(f"{ROOT_DIR}/{IMAGE_FOLDER}/{image.filename}", 'wb') as buffer:
            # print(buffer)
            shutil.copyfileobj(image.file, buffer)
        image_classifier = evaluate_image(path)
        remove_file_if_exists(path)
        return image_classifier
    else:
        return "File Format is not Compatible"




if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)