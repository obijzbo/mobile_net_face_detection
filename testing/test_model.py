import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from configs.config_ml_model import BATCH_SIZE, CLASSIFICATION_REPORT_PATH, ROOT_DIR, MODEL_PATH
from functions import save_classification


def test(data_preprocess, data_len, dir_name):
	model = load_model(f"{dir_name}/{MODEL_PATH}/model.h5")
	test = data_preprocess["test_Gen"]
	test.reset()
	pred_Idxs = model.predict(x=test, steps=(data_len["total_test"] // BATCH_SIZE) + 1)

	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	pred_Idxs = np.argmax(pred_Idxs, axis=1)

	# show a nicely formatted classification report
	report = classification_report(test.classes, pred_Idxs,
		target_names=test.class_indices.keys())

	save_classification(report, dir_name)