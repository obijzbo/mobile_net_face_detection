import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense,BatchNormalization, Flatten, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from configs.config_ml_model import EPOCHS, BATCH_SIZE
from functions import save_model_summary, save_model, save_accuracy, save_plot

def train(data_preprocess, data_len, dir_name):
    mobileNet = MobileNetV2(include_top = False, weights = "imagenet" ,input_shape=(96,96,3))

    model = Sequential([
        mobileNet,
        GlobalAveragePooling2D(),
        Dense(300, activation=tf.nn.leaky_relu),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    summary = model.summary()
    save_model_summary(model, dir_name)


    histry = model.fit(
        x = data_preprocess["train_Gen"],
        steps_per_epoch = data_len["total_train"] // BATCH_SIZE,
        validation_data = data_preprocess["validation_Gen"],
        validation_steps = data_len["total_val"] // BATCH_SIZE,
        epochs=EPOCHS)
    save_model(model, dir_name)
    save_accuracy(histry, dir_name)
    save_plot(EPOCHS, histry, dir_name)