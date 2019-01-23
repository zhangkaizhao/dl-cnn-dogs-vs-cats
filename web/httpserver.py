# -*- coding: utf-8 -*-
import asyncio
from concurrent import futures
from io import BytesIO
import json
import os.path

from PIL import Image
from aiohttp import web
from keras.models import load_model
import numpy as np

from .prediction import predict_classes

MAX_THREAD_WORKERS = 5
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MiB
IMAGE_TARGET_SIZE = (150, 150)

executor = futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS)

_here = os.path.abspath(os.path.dirname(__file__))
home_filepath = os.path.join(_here, "static", "index.html")


def is_cat(value_to_class, predicted_classes):
    class_value = predicted_classes[0]
    return value_to_class.get(class_value) == "cats"


async def home(request):
    return web.FileResponse(home_filepath)


async def new_file(request):
    """Starting predict image"""
    reader = await request.multipart()
    field = await reader.next()
    if field.name != "image":
        raise web.HTTPBadRequest(text="no file in 'image' field uploaded")

    buf = BytesIO()
    file_size = 0
    while True:
        chunk = await field.read_chunk()
        if not chunk:
            break

        next_file_size = file_size + len(chunk)
        if next_file_size > MAX_FILE_SIZE:
            raise web.HTTPBadRequest(text="max file size 1 MiB")

        buf.write(chunk)
        file_size = next_file_size

    try:
        img = Image.open(buf)
    except OSError:
        raise web.HTTPBadRequest(text="only support JPEG format")

    if img.format.upper() != "JPEG":
        raise web.HTTPBadRequest(text="only support JPEG format")

    model = request.app["cnn_model"]
    value_to_class = request.app["cnn_model_value_to_class"]

    loop = asyncio.get_event_loop()
    predicted_classes = await loop.run_in_executor(
        executor,
        predict_classes,
        model,
        buf,
    )

    buf.close()

    if is_cat(value_to_class, predicted_classes):
        text = u"是猫"
    else:
        text = u"不是猫"

    return web.Response(text=text)


def init(cnn_network):
    app = web.Application()

    model_filepath = "models/model_{}.h5".format(cnn_network)
    model = load_model(model_filepath)
    # https://github.com/keras-team/keras/issues/6462#issuecomment-319232504
    model._make_predict_function()
    app["cnn_model"] = model

    classes_filepath = "models/classes_{}.json".format(cnn_network)
    with open(classes_filepath, "r") as f:
        classes = json.load(f)

    value_to_class = {v: k for (k, v) in classes.items()}
    app["cnn_model_value_to_class"] = value_to_class

    app.add_routes([web.get("/", home)])
    app.add_routes([web.post("/new_file", new_file)])

    return app


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="cats vs dogs server")
    parser.add_argument("--path")
    parser.add_argument("--port")
    parser.add_argument("--network")
    args = parser.parse_args()

    cnn_network = args.network or "lenet5"
    if cnn_network not in ["lenet5", "vgg16"]:
        sys.exit(1)
    else:
        print("using cnn network: {}".format(cnn_network))

    app = init(cnn_network)

    web.run_app(app, path=args.path, port=args.port)
