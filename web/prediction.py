from keras.models import load_model
from keras.preprocessing import image
import numpy as np


def load_image(filepath):
    img = image.load_img(filepath, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor


def predict_classes(model, image_filepath):
    img_tensor = load_image(image_filepath)
    return model.predict_classes(img_tensor)


if __name__ == "__main__":
    import json
    import os
    import os.path
    import random

    model = load_model("models/model_lenet5.h5")

    with open("models/classes_lenet5.json", "r") as f:
        classes = json.load(f)

    value_to_class = {v: k for (k, v) in classes.items()}

    cat_img_filepaths = [
        os.path.join("data/validation/cats", path)
        for path in random.sample(os.listdir("data/validation/cats"), 10)
    ]
    dog_img_filepaths = [
        os.path.join("data/validation/dogs", path)
        for path in random.sample(os.listdir("data/validation/dogs"), 10)
    ]

    for class_, img_filepaths in [
        ("cats", cat_img_filepaths),
        ("dogs", dog_img_filepaths),
    ]:
        for img_filepath in img_filepaths:
            predicted_classes = predict_classes(model, img_filepath)
            print(img_filepath)
            class_value = predicted_classes[0]
            is_cat = value_to_class.get(class_value) == "cats"
            print("cat? {}".format(is_cat))
            print("")
        print("")
