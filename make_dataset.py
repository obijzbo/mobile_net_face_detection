import os
import random

from configs.config_ml_model import ROOT_DIR
from configs.config_data import CURRENT_DATA_SET, TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH
from functions import shuffle_folder_items, make_train_test_val, copy_item_to_new_folder



dir = f"{ROOT_DIR}/{CURRENT_DATA_SET}"
print(dir)
categories = os.listdir(f"{ROOT_DIR}/{CURRENT_DATA_SET}")
for category in categories:
    path = f"{dir}/{category}"
    # items = os.listdir(path)
    # items = random.shuffle(items)
    # print(items)
    # print(len(items))
    category_items = shuffle_folder_items(path)
    print(category_items)
    make_train_test_val(category_items, category)