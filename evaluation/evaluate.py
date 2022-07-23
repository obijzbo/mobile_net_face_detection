from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import os
import cv2
from tensorflow.keras.models import load_model
from configs.config_data import TEST_DATA_PATH, DATA_PATH
from configs.config_ml_model import ROOT_DIR, MODEL_VERSION_IN_WORK, MODEL_PATH, RESOURCE_PATH
from functions import img_process, create_montages, save_classification, draw_plot_confusion_matrix, get_latest_version


def evaluation(dir_name):
    model = load_model(f"{ROOT_DIR}/{RESOURCE_PATH}/{MODEL_VERSION_IN_WORK}/{MODEL_PATH}/model.h5")
    categories_folder = f"{ROOT_DIR}/{DATA_PATH}/{TEST_DATA_PATH}"
    category_true = []
    category_pred = []
    results = []
    categories = os.listdir(categories_folder)
    for category in categories:
        image_path = f"{categories_folder}/{category}"
        image_list = os.listdir(f"{categories_folder}/{category}")
        print(image_list)
        for image in image_list:
            org = cv2.imread(f"{image_path}/{image}")
            image = img_process(org)
            # make predictions on the input image
            pred = model.predict(image)
            pred = pred.argmax(axis=1)[0]
            if category == "real":
                category_true.append(1)
            elif category == "fake":
                category_true.append(0)
            label = "Not Face" if pred == 0 else "Face"
            color = (0, 0, 255) if pred == 0 else (0, 255, 0)
            org = cv2.resize(org, (128, 128))
            cv2.putText(org, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)
            results.append(org)

            category_pred.append(pred)

    classification_accuracy = metrics.classification_report(category_true, category_pred)
    # accuracy = metrics.accuracy_score(category_true, category_pred)
    # precision_positive = metrics.precision_score(category_true, category_pred, pos_label=1)
    # precision_negative = metrics.precision_score(category_true, category_pred, pos_label=0)
    # recall_sensitivity = metrics.recall_score(category_true, category_pred, pos_label=1)
    # recall_specificity = metrics.recall_score(category_true, category_pred, pos_label=0)

    save_classification(classification_accuracy, dir_name)

    data = confusion_matrix(category_true, category_pred)
    draw_plot_confusion_matrix(data, categories, dir_name)

    create_montages(results, 8, 96, dir_name)



def evaluate_image(img_path):
    file_path = f"{ROOT_DIR}/{RESOURCE_PATH}"
    latest_version = get_latest_version(file_path)
    model = load_model(f"{file_path}/{latest_version}/{MODEL_PATH}/model.h5")
    org = cv2.imread(f"{ROOT_DIR}/{img_path}")
    image = img_process(org)
    pred = model.predict(image)
    pred = pred.argmax(axis=1)[0]
    if pred == 0:
        return "Not Face"
    elif pred == 1:
        return "Face"