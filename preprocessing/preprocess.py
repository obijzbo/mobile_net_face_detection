from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configs.config_data import TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH, DATA_PATH
from configs.config_ml_model import ROOT_DIR
from configs.config_ml_model import EPOCHS, BATCH_SIZE

def preprocess():
    train_path = f"{ROOT_DIR}/{DATA_PATH}/{TRAIN_DATA_PATH}"
    test_path = f"{ROOT_DIR}/{DATA_PATH}/{TEST_DATA_PATH}"
    validation_path = f"{ROOT_DIR}/{DATA_PATH}/{VALIDATION_DATA_PATH}"

    totalTrain = len(list(paths.list_images(train_path)))
    totalVal = len(list(paths.list_images(validation_path)))
    totalTest = len(list(paths.list_images(test_path)))

    train_augmentation = ImageDataGenerator(rescale=1 / 255.0,
                                            rotation_range=20,
                                            zoom_range=0.05,
                                            width_shift_range=0.05,
                                            height_shift_range=0.05,
                                            shear_range=0.05,
                                            horizontal_flip=True,
                                            fill_mode="nearest", )

    val_augmentation = ImageDataGenerator(rescale=1 / 255.0)

    # initialize the training generator
    train = train_augmentation.flow_from_directory(
        train_path,
        class_mode="categorical",
        target_size=(96, 96),
        color_mode="rgb",
        shuffle=True,
        batch_size=BATCH_SIZE)

    test = val_augmentation.flow_from_directory(
        test_path,
        class_mode="categorical",
        target_size=(96, 96),
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE,
    )

    # initialize the validation generator
    val = val_augmentation.flow_from_directory(
        validation_path,
        class_mode="categorical",
        target_size=(96, 96),
        color_mode="rgb",
        shuffle=False,
        batch_size=BATCH_SIZE,
    )

    data_preprocess = {"train_Gen": train,
                       "test_Gen": test,
                       "validation_Gen": val}

    data_len = {"total_train" : totalTrain,
                "total_test" : totalTest,
                "total_val" : totalVal}

    return data_preprocess, data_len