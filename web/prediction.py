from keras.models import load_model
from keras.preprocessing import image
import numpy as np


def load_image(filepath):
    img = image.load_img(filepath, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor


def predict(model, image_filepath):
    img_tensor = load_image(image_filepath)
    return model.predict(img_tensor)


if __name__ == "__main__":
    import json
    import os
    import os.path
    import random

    model = load_model("models/model_lenet5.h5")

    with open("models/classes_lenet5.json", "r") as f:
        classes = json.load(f)

    cat_is_smaller = classes["cats"] < classes["dogs"]

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
            prediction = predict(model, img_filepath)
            print(img_filepath)
            print(prediction)
            result_is_smaller = prediction[0][0] < prediction[0][1]
            if cat_is_smaller and result_is_smaller:
                is_cat = True
            elif (not cat_is_smaller) and (not result_is_smaller):
                is_cat = True
            else:
                is_cat = False
            # is_cat = (cat_is_smaller + result_is_smaller) != 1
            print("cat? {}".format(is_cat))
            print("")
        print("")
