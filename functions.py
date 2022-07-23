import json
import os
import numpy as np
import pandas as pd
import seaborn as sn
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import build_montages
from matplotlib import pyplot as plt
import cv2
import shutil

from configs.config_ml_model import ROOT_DIR, RESOURCE_PATH, MODEL_SUMMARY_PATH, MODEL_PATH, ACCURACY_PATH, PLOT_PATH, CLASSIFICATION_REPORT_PATH, EVALUATION_REPORT, MONTAGES
from configs.config_data import TRAIN_DATA_PATH,TEST_DATA_PATH,VALIDATION_DATA_PATH, CURRENT_DATA_SET, DATA_PATH



def make_version(file_path):
    try:
        existing_versions = os.listdir(file_path)
        versions = [int(v) for v in existing_versions]
        if versions:
            last_version = max(versions)
            new_version = last_version + 1
        else:
            new_version = 1
    except FileNotFoundError as e:
        print(e)
        new_version = 1
    print(f'Version : {new_version}')
    return new_version


def get_latest_version(file_path):
    try:
        existing_versions = os.listdir(file_path)
        versions = [int(v) for v in existing_versions]
        latest_version = max(versions)
        return latest_version
    except FileNotFoundError as e:
        print(e)



def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created.")
    except FileExistsError:
        print("Directory ", dir_name, " already exist.")


# check directory if not exists then make directory
def make_dir_if_not_exists(file_path):
    dirs = file_path.split('/')
    if dirs:
        path = ''
        for dir in dirs:
            if dir:
                path = path + dir + '/'
                if not os.path.exists(path):
                    os.mkdir(path)



# remove file if exists
def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} file successfully removed!')
    else:
        print(f'{file_path} path not exists!')



def save_model_summary(model, dir_name):
    dir_name = f"{dir_name}/{MODEL_SUMMARY_PATH}"
    make_dir(dir_name)
    with open(f"{dir_name}/modelSummary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def save_model(model, dir_name):
    dir_name = f"{dir_name}/{MODEL_PATH}"
    make_dir(dir_name)
    model.save(f"{dir_name}/model.h5")
    return dir_name


def save_accuracy(history, dir_name):
    dir_name = f"{dir_name}/{ACCURACY_PATH}"
    make_dir(dir_name)
    accuracy_report = {"train_accuracy" : sum(history.history["accuracy"])/len(history.history["accuracy"]),
                       "train_loss" : sum(history.history["loss"])/len(history.history["loss"]),
                       "validation_accuracy" : sum(history.history["val_accuracy"])/len(history.history["val_accuracy"]),
                       "validation_loss" : sum(history.history["val_loss"])/len(history.history["val_loss"])}
    with open(f"{dir_name}/accuracy_report.json", "w") as f:
        json.dump(accuracy_report, f, indent=4)


def save_classification(report, dir_name):
    dir_name = f"{dir_name}/{CLASSIFICATION_REPORT_PATH}"
    make_dir(dir_name)
    with open(f'{dir_name}/classification_report.csv', 'w') as f:
        f.write(report)


def save_validation(img_num, label_count, root_dir, validation_path):
    folder_name = make_version(f"{root_dir}/{RESOURCE_PATH}/{validation_path}")
    dir_name = f"{root_dir}/{RESOURCE_PATH}/{validation_path}/{folder_name}"
    make_dir(dir_name)
    validation_report = {"Number of Images":img_num,
                         "Human Face":label_count["Face"],
                         "Non Human Face":label_count["Not_Face"]}
    with open(f"{dir_name}/validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=4)


def save_montage_img(i, montage, dir_name):
    dir_name = f"{dir_name}/{MONTAGES}"
    make_dir(dir_name)
    try:
        cv2.imwrite(f"{dir_name}/montage{i}.jpg",montage)
        print(f"Saved image at : {dir_name}")
    except Exception as e:
        print(f"ERROR - Could not save at : {dir_name}")


def draw_plot(epoch, history, dir_name):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epoch), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epoch), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epoch), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(f"{dir_name}/plot")


def save_plot(epoch, history, dir_name):
    dir_name = f"{dir_name}/{PLOT_PATH}"
    make_dir(dir_name)
    draw_plot(epoch, history, dir_name)


def draw_plot_confusion_matrix(data, categories, dir_name):
    df_cm = pd.DataFrame(data, columns=categories, index=categories)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, fmt='d')  # font size
    dir_name = f"{dir_name}/{PLOT_PATH}"
    make_dir(dir_name)
    plt.savefig(f"{dir_name}/plot")
    plt.show()


def create_montages(results, grid_size, img_size, dir_name):
    montages = build_montages(results, (img_size, img_size), (grid_size, grid_size))
    for i, montage in enumerate(montages):
        cv2.imshow("Result", montage)
        save_montage_img(i, montage, dir_name)
        cv2.waitKey(5000)


def img_process(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0

    # order channel dimensions (channels-first or channels-last)
    # depending on our Keras backend, then add a batch dimension to
    # the image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return(image)


def shuffle_folder_items(dir):
    items = os.listdir(dir)
    return items


def make_train_test_val(shuffled_items, category):
        category_len = len(shuffled_items)
        train_end = int(category_len*0.7)
        val_end = int(category_len*0.2)
        test_end = int(category_len*0.1)
        train_list = shuffled_items[:(train_end-1)]
        val_list = shuffled_items[train_end:(train_end+val_end)-1]
        test_list = shuffled_items[val_end:(val_end+test_end)-1]
        copy_item_to_new_folder(train_list, f"{CURRENT_DATA_SET}/{category}", f"{ROOT_DIR}/{DATA_PATH}/{TRAIN_DATA_PATH}/{category}")
        copy_item_to_new_folder(val_list, f"{CURRENT_DATA_SET}/{category}", f"{ROOT_DIR}/{DATA_PATH}/{VALIDATION_DATA_PATH}/{category}")
        copy_item_to_new_folder(test_list, f"{CURRENT_DATA_SET}/{category}", f"{ROOT_DIR}/{DATA_PATH}/{TEST_DATA_PATH}/{category}")



def copy_item_to_new_folder(item_list, from_dir, to_dir):
    for item in item_list:
        try:
            shutil.copy(f"{from_dir}/{item}",f"{to_dir}")
            print(f"{item} copied successfully")
        except shutil.SameFileError:
            print("Source and destination represents the same file")