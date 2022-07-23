from evaluation.evaluate import evaluation
from configs.config_ml_model import ROOT_DIR, EVALUATION_REPORT
from functions import make_version, make_dir


version = make_version(f"{ROOT_DIR}/{EVALUATION_REPORT}")
dir_name = f"{ROOT_DIR}/{EVALUATION_REPORT}/{version}"
make_dir(dir_name)
evaluation(dir_name)