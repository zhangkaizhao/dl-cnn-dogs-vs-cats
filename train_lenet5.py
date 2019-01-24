"""
LeNet-5 is a classic convolutional network.

LeNet-5, a pioneering 7-level convolutional network by LeCun et al. in 1998,
that classifies digits, was applied by several banks to recognize hand-written
numbers on checks (British English: cheques) digitized in 32x32 pixel images.
The ability to process higher resolution images requires larger and more layers
of convolutional neural networks, so this technique is constrained by the
availability of computing resources.
(via https://en.wikipedia.org/wiki/Convolutional_neural_network#LeNet-5 )

For more details about LeNet-5:
https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
https://www.jiqizhixin.com/articles/2016-09-08-2 (Chinese translation)

In this script our model uses LeNet-5 network.

Hidden layers used in order:
- 3 convolution layers and 3 pooling layers (subsampling layers):
  - C1:
    - filters: 32
    - kerne_size: (3, 3)
    - activation: relu
  - S2 (max pooling):
    - pool_size: (2, 2)
  - C3:
    - filters: 32
    - kerne_size: (3, 3)
    - activation: relu
  - S4 (max pooling):
    - pool_size: (2, 2)
  - C5:
    - filters: 32
    - kerne_size: (3, 3)
    - activation: relu
  - S6 (max pooling):
    - pool_size: (2, 2)
- 2 full connection layers:
  - Dropout
    - rate: 0.25
  - Flatten
  - Dense
    - units: 64
    - activation: relu
  - Dropout
    - rate: 0.5
  - Dense
    - units: 2
    - activation: softmax

And we use SGD (stochastic gradient descent) optimizer for optimization.

Finally we use `EarlyStopping` callback to stop training when a monitored
quantity ("val_loss" here) has stopped improving.
"""
import json

from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMCallback

from settings import BATCH_SIZE, TARGET_SIZE, TRAIN_RATIO, TRAIN_TOTAL

# #####################################
# image data augumentation generators
# #####################################

# `ImageDataGenerator` is used to generate batches of tensor image data with
# real-time data augmentation. The data will be looped over (in batches).
#
# ```python
# keras.preprocessing.image.ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     zca_epsilon=1e-06,
#     rotation_range=0,
#     width_shift_range=0.0,
#     height_shift_range=0.0,
#     brightness_range=None,
#     shear_range=0.0,
#     zoom_range=0.0,
#     channel_shift_range=0.0,
#     fill_mode='nearest',
#     cval=0.0,
#     horizontal_flip=False,
#     vertical_flip=False,
#     rescale=None,
#     preprocessing_function=None,
#     data_format=None,
#     validation_split=0.0,
#     dtype=None)
# ```
#
# Arguments:
# - rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is
#            applied, otherwise we multiply the data by the value provided
#            (after applying all other transformations).
# - shear_range: Float. Shear Intensity (Shear angle in counter-clockwise
#                direction in degrees)
# - zoom_range: Float or [lower, upper]. Range for random zoom. If a float,
#               `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
# - horizontal_flip: Boolean. Randomly flip inputs horizontally.
#
# For more arguments: https://keras.io/preprocessing/image/#imagedatagenerator-class .

# image data augmentation generator for training
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# image data augmentation generator for validation
val_datagen = ImageDataGenerator(rescale=1/255.0)

# #################
# file generators
# #################

# `flow_from_directory` method of image data augmentation generator takes the
# path to a directory & generates batches of augmented data.
#
# The return is a `DirectoryIterator` yielding tuples of `(x, y)` where `x` is
# a numpy array containing a batch of images with shape
# `(batch_size, *target_size, channels)` and `y` is a numpy array of
# corresponding labels.
#
# ```python
# keras.preprocessing.image.ImageDataGenerator.flow_from_directory(
#     directory,
#     target_size=(256, 256),
#     color_mode='rgb',
#     classes=None,
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=None,
#     save_to_dir=None,
#     save_prefix='',
#     save_format='png',
#     follow_links=False,
#     subset=None,
#     interpolation='nearest')
# ```
#
# Arguments:
# - directory: Path to the target directory. It should contain one subdirectory
#              per class. Any PNG, JPG, BMP, PPM or TIF images inside each of
#              the subdirectories directory tree will be included in the
#              generator. See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
#              for more details.
# - target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
#                The dimensions to which all images found will be resized.
# - batch_size: size of the batches of data (default: 32).
# - class_mode: one of "categorical", "binary", "sparse", "input", "other" or
#               None. Default: "categorical".
#               Determines the type of label arrays that are returned:
#               - `"categorical"` will be 2D one-hot encoded labels,
#               - `"binary"` will be 1D binary labels,
#               - `"sparse"` will be 1D integer labels,
#               - `"input"` will be images identical
#
# For more arguments: https://keras.io/preprocessing/image/#flow_from_directory .

# file generator for training
train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# file generator for validation
validation_generator = val_datagen.flow_from_directory(
    "data/validation",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# ##############
# model desgin
# ##############

# The Sequential model is a linear stack of layers.

model = Sequential()

# Here we will use the the `.add()` method to add layers to the model.

# 3 convolution layers and 3 pooling layers.

# As we have set the `class_mode` to `"categorical"` for image data
# augmentation generators, we will use 2D convolutional layers.

# For max pooling shape, typical values are 2×2. Very large input volumes may
# warrant 4×4 pooling in the lower layers. However, choosing larger shapes will
# dramatically reduce the dimension of the signal, and may result in excess
# information loss. Often, non-overlapping pooling windows perform best.
# (via https://en.wikipedia.org/wiki/Convolutional_neural_network#Max_pooling_shape )

# `Conv2D` is 2D convolution layer (e.g. spatial convolution over images).
#
# ```python
# keras.layers.Conv2D(
#     filters,
#     kernel_size,
#     strides=(1, 1),
#     padding='valid',
#     data_format=None,
#     dilation_rate=(1, 1),
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None)
# ```
#
# This layer creates a convolution kernel that is convolved with the layer
# input to produce a tensor of outputs. If use_bias is True, a bias vector is
# created and added to the outputs. Finally, if activation is not None, it is
# applied to the outputs as well.
#
# When using this layer as the first layer in a model, provide the keyword
# argument `input_shape` (tuple of integers, does not include the batch axis),
# e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures in
# `data_format="channels_last"`.
#
# Arguments:
# - filters: Integer, the dimensionality of the output space
#            (i.e. the number of output filters in the convolution).
# - kernel_size: An integer or tuple/list of 2 integers, specifying the height
#                and width of the 2D convolution window. Can be a single
#                integer to specify the same value for all spatial dimensions.
# - strides: An integer or tuple/list of 2 integers, specifying the strides of
#            the convolution along the height and width. Can be a single
#            integer to specify the same value for all spatial dimensions.
#            Specifying any stride value != 1 is incompatible with specifying
#            any `dilation_rate` value != 1.
# - padding: one of `"valid"` or `"same"` (case-insensitive).
#            Note that `"same"` is slightly inconsistent across backends with
#            `strides` != 1, as described [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
# - data_format: A string, one of `"channels_last"` (default) or
#                `"channels_first"`. The ordering of the dimensions in the
#                inputs. `"channels_last"` corresponds to inputs with shape
#                `(batch, steps, channels)` (default format for temporal data
#                in Keras) while `"channels_first"` corresponds to inputs with
#                shape `(batch, channels, steps)`.
# - activation: Activation function to use
#               (see [activations](https://keras.io/activations/)).
#               If you don't specify anything, no activation is applied
#               (ie. "linear" activation: `a(x) = x`).
#
# For more arguments: https://keras.io/layers/convolutional/ .

# `MaxPooling2D` is max pooling operation for spatial data.
#
# ```python
# keras.layers.MaxPooling2D(
#     pool_size=(2, 2),
#     strides=None,
#     padding='valid',
#     data_format=None)
# ```
#
# Arguments:
# - pool_size: integer or tuple of 2 integers, factors by which to downscale
#              (vertical, horizontal). (2, 2) will halve the input in both
#              spatial dimension. If only one integer is specified, the same
#              window length will be used for both dimensions.
#
# For more arguments: https://keras.io/layers/pooling/ .

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

# 2 full connection layers.

# Because a fully connected layer occupies most of the parameters, it is prone
# to overfitting. One method to reduce overfitting is dropout.

# We use Dropout to reduce overfitting here.

# `Dense` is just your regular densely-connected NN layer.
#
# ```python
# keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None)
# ```
#
# Dense implements the operation:
# `output = activation(dot(input, kernel) + bias)` where `activation` is the
# element-wise activation function passed as the `activation argument`,
# `kernel` is a weights matrix created by the layer, and `bias` is a bias
# vector created by the layer (only applicable if `use_bias` is `True`).
#
# Arguments:
# - units: Positive integer, dimensionality of the output space.
# - activation: Activation function to use
#               (see [activations](https://keras.io/activations/)).
#               If you don't specify anything, no activation is applied
#               (ie. "linear" activation: `a(x) = x`).
#
# For more arguments: https://keras.io/layers/core/#dense .

# `Dropout` applies Dropout to the input.
#
# ```python
# keras.layers.Dropout(
#     rate,
#     noise_shape=None,
#     seed=None)
# ```
#
# Dropout consists in randomly setting a fraction rate of input units to 0 at
# each update during training time, which helps prevent overfitting.
#
# Arguments:
# - rate: float between 0 and 1. Fraction of the input units to drop.
#
# For more arguments: https://keras.io/layers/core/#dropout .

# `Flatten` flattens the input. Does not affect the batch size.
#
# ```python
# keras.layers.Flatten(data_format=None)
# ```
#
# For more arguments: https://keras.io/layers/core/#flatten .

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))


# optimization

# We use SGD optimizer for optimization.

#
# `SGD` is stochastic gradient descent optimizer.
#
# Includes support for momentum, learning rate decay, and Nesterov momentum.
#
# ```python
# keras.optimizers.SGD(
#     lr=0.01,
#     momentum=0.0,
#     decay=0.0,
#     nesterov=False)
# ```python
#
# Arguments:
# - lr: float >= 0. Learning rate.
# - momentum: float >= 0. Parameter that accelerates SGD in the relevant
#             direction and dampens oscillations.
# - decay: float >= 0. Learning rate decay over each update.
# - nesterov: boolean. Whether to apply Nesterov momentum.
#
# via https://keras.io/optimizers/#sgd

#
# The `compile` method of model is used to configures the model for training.
#
# ```python
# keras.models.Model.compile(
#     optimizer,
#     loss=None,
#     metrics=None,
#     loss_weights=None,
#     sample_weight_mode=None,
#     weighted_metrics=None,
#     target_tensors=None)
# ```
#
# Arguments:
# - loss: String (name of objective function) or objective function.
#         See [losses](https://keras.io/losses).
#         If the model has multiple outputs, you can use a different loss on
#         each output by passing a dictionary or a list of losses. The loss
#         value that will be minimized by the model will then be the sum of all
#         individual losses.
# - optimizer: String (name of optimizer) or optimizer instance.
#              See [optimizers](https://keras.io/optimizers).
# - metrics: List of metrics to be evaluated by the model during training and
#            testing. Typically you will use `metrics=['accuracy']`. To specify
#            different metrics for different outputs of a multi-output model,
#            you could also pass a dictionary, such as
#            `metrics={'output_a': 'accuracy'}`.
#
# For more arguments: https://keras.io/models/model/#compile

EPOCHS = 50
LRATE = 0.01
DECAY = LRATE / EPOCHS
sgd = SGD(lr=LRATE, momentum=0.9, decay=DECAY, nesterov=False)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.summary()

# ####################
# callback functions
# ####################


class LossHistory(Callback):
    """Callback for loss logging per epoch"""
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


history = LossHistory()

# `EarlyStopping` stops training when a monitored quantity has stopped
# improving.
#
# ```python
# keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=0,
#     patience=0,
#     verbose=0,
#     mode='auto',
#     baseline=None,
#     restore_best_weights=False)
# ```
#
# Arguments:
# - monitor: quantity to be monitored.
# - min_delta: minimum change in the monitored quantity to qualify as an
#              improvement, i.e. an absolute change of less than min_delta,
#              will count as no improvement.
# - patience: number of epochs with no improvement after which training will
#             be stopped.
# - verbose: verbosity mode.
# - mode: one of {auto, min, max}.
#         - In `min` mode, training will stop when the quantity monitored has
#           stopped decreasing;
#         - in `max` mode it will stop when the quantity monitored has stopped
#           increasing;
#         - in `auto` mode, the direction is automatically inferred from the
#           name of the monitored quantity.
#
# For more arguments: https://keras.io/callbacks/#EarlyStopping .

# Callback for early stopping the training
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
)


# ##########
# training
# ##########

# `fit_generator` method trains the model on data generated batch-by-batch by
# a Python generator (or an instance of `Sequence`).
#
# ```python
# keras.models.Model.fit_generator(
#     generator,
#     steps_per_epoch=None,
#     epochs=1,
#     verbose=1,
#     callbacks=None,
#     validation_data=None,
#     validation_steps=None,
#     class_weight=None,
#     max_queue_size=10,
#     workers=1,
#     use_multiprocessing=False,
#     shuffle=True,
#     initial_epoch=0)
# ```
#
# The generator is run in parallel to the model, for efficiency. For instance,
# this allows you to do real-time data augmentation on images on CPU in
# parallel to training your model on GPU.
#
# Arguments:
# - generator: A generator or an instance of `Sequence`
#              (`keras.utils.Sequence`) object in order to avoid duplicate data
#              when using multiprocessing. The output of the generator must be
#              either
#              - a tuple (inputs, targets)
#              - a tuple (inputs, targets, sample_weights).
# - steps_per_epoch: Integer. Total number of steps (batches of samples) to
#                    yield from `generator` before declaring one epoch finished
#                    and starting the next epoch. It should typically be equal
#                    to the number of samples of your dataset divided by the
#                    batch size. Optional for `Sequence`: if unspecified, will
#                    use the `len(generator)` as a number of steps.
# - epochs: Integer. Number of epochs to train the model. An epoch is an
#           iteration over the entire data provided, as defined by
#           `steps_per_epoch`. Note that in conjunction with `initial_epoch`,
#           `epochs` is to be understood as "final epoch". The model is not
#           trained for a number of iterations given by epochs, but merely
#           until the epoch of index `epochs` is reached.
# - validation_data: This can be either
#                    - a generator or a `Sequence` object for the validation
#                      data
#                    - tuple `(x_val, y_val)`
#                    - tuple `(x_val, y_val, val_sample_weights)`
#                    on which to evaluate the loss and any model metrics at the
#                    end of each epoch. The model will not be trained on this
#                    data.
# - validation_steps: Only relevant if `validation_data` is a generator. Total
#                     number of steps (batches of samples) to yield from
#                     `validation_data` generator before stopping at the end of
#                     every epoch. It should typically be equal to the number
#                     of samples of your validation dataset divided by the
#                     batch size. Optional for `Sequence`: if unspecified, will
#                     use the `len(validation_data)` as a number of steps.
# - callbacks: List of `keras.callbacks.Callback` instances. List of callbacks
#              to apply during training.
#              See [callbacks](https://keras.io/callbacks).
# - verbose: Integer. 0, 1, or 2. Verbosity mode.
#            0 = silent, 1 = progress bar, 2 = one line per epoch.
#
# For more arguments: https://keras.io/models/model/#fit_generator .

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

# ####################
# save trained model
# ####################

model.save("models/model_lenet5.h5")

# save classes
classes = train_generator.class_indices
with open("models/classes_lenet5.json", "w") as f:
    json.dump(classes, f, indent=2)
