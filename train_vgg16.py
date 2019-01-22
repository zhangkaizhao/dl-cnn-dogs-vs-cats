import json

from keras import applications
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMCallback
import numpy as np

from settings import BATCH_SIZE, TARGET_SIZE, TRAIN_RATIO, TRAIN_TOTAL


# include_top: whether to include the 3 full-connected layers
# at the top of the network
model = applications.VGG16(include_top=False, weights="imagenet")
datagen = ImageDataGenerator(rescale=1.0/255)

#

train_generator = datagen.flow_from_directory(
    "data/train",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)

bottleneck_features_train = model.predict_generator(
    train_generator,
    int(TRAIN_TOTAL * (1 - TRAIN_RATIO)) // BATCH_SIZE,
)
with open("features/bottleneck_features_train.npy", "wb") as bft:
    np.save(bft, bottleneck_features_train)

validation_generator = datagen.flow_from_directory(
    "data/validation",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)

bottleneck_features_validation = model.predict_generator(
    validation_generator,
    int(TRAIN_TOTAL * (1 - TRAIN_RATIO)) // BATCH_SIZE,
)
with open("features/bottleneck_features_validation.npy", "wb") as bft:
    np.save(bft, bottleneck_features_validation)

#

train_data = np.load("features/bottleneck_features_train.npy")
train_labels = np.array(
    [0] * (int((1 - TRAIN_RATIO) * TRAIN_TOTAL) // 2) +
    [1] * (int((1 - TRAIN_RATIO) * TRAIN_TOTAL) // 2)
)

validation_data = np.load("features/bottleneck_features_validation.npy")
validation_labels = np.array(
    [0] * (int(TRAIN_RATIO * TRAIN_TOTAL) // 2) +
    [1] * (int(TRAIN_RATIO * TRAIN_TOTAL) // 2)
)

# callback functions


class LossHistory(Callback):
    """Callback for loss logging per epoch"""
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


history = LossHistory()

# model

model = Sequential()
model.add(Flatten(input_shape=train_data[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="sigmoid"))

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

fitted_model = model.fit(
    train_data,
    train_labels,
    epochs=2,  # 15
    batch_size=BATCH_SIZE,
    validation_data=(
        validation_data,
        validation_labels[:validation_data.shape[0]]
    ),
    verbose=0,
    callbacks=[TQDMCallback(leave_inner=True, leave_outer=False), history],
)

# save model

model.save("models/model_vgg16.h5")

# save classes
classes = train_generator.class_indices
with open("models/classes_vgg16.json", "w") as f:
    json.dump(classes, f, indent=2)
