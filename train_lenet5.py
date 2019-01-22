import json

from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMCallback

from settings import BATCH_SIZE, TARGET_SIZE, TRAIN_RATIO, TRAIN_TOTAL


# image data generators

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1/255.0)


# file generators

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

validation_generator = val_datagen.flow_from_directory(
    "data/validation",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)


# model

# 3 conv filters and 3 pooling layers
# 2 full connection layers
# conv filter: 32x32x64 3*3

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        input_shape=(TARGET_SIZE + (3,)),
        padding="same",
        activation="relu",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))


# optimize

EPOCHS = 50
LRATE = 0.01
DECAY = LRATE / EPOCHS
sgd = SGD(lr=LRATE, momentum=0.9, decay=DECAY, nesterov=False)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.summary()


# networking


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

# Callback for early stopping the training
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
)


# training

fitted_model = model.fit_generator(
    train_generator,
    steps_per_epoch=int(TRAIN_TOTAL * (1 - TRAIN_RATIO) / BATCH_SIZE),
    epochs=8,  # 50
    validation_data=validation_generator,
    validation_steps=int(TRAIN_TOTAL * TRAIN_RATIO) // BATCH_SIZE,
    callbacks=[
        TQDMCallback(leave_inner=True, leave_outer=True),
        early_stopping,
        history,
    ],
    verbose=0,
)


# save model

model.save("models/model_lenet5.h5")

# save classes
classes = train_generator.class_indices
with open("models/classes_lenet5.json", "w") as f:
    json.dump(classes, f, indent=2)
