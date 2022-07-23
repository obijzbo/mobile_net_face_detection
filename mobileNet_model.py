from functions import make_version, make_dir
from configs.config_ml_model import ROOT_DIR, RESOURCE_PATH
from preprocessing.preprocess import preprocess
from training.train import train
from testing.test_model import test

version = make_version(f"{ROOT_DIR}/{RESOURCE_PATH}")
dir_name = f"{ROOT_DIR}/{RESOURCE_PATH}/{version}"
make_dir(dir_name)

data_preprocess, data_len = preprocess()

# print(data_preprocess)
# print(data_len)

train(data_preprocess, data_len, dir_name)
test(data_preprocess, data_len, dir_name)