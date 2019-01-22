# Dogs vs. Cats in Deep Learning CNN

data: https://www.kaggle.com/c/dogs-vs-cats/data

Support Python 2.7 and Python 3.6 (tensoflow does not support Python 3.7 at the moment).

> Warning: This is a draft for demo.

## Preparation

### Download and extract source data

Download full data from data source. Then extract `train.zip` to `all/` directory.
All the image files for training should be under `all/train/` directory.

### Setup virutualenv

Create a virtual env:

```sh
virtualenv venv
```

or

```sh
python3.6 -m venv venv
```

for Python 3.6.

And then activate it:

```sh
source venv/bin/activate
```

### Install dependencies

Use `pip` to install dependencies for this virtual env:

```
pip install -r requirements.txt
```

If any issues happened, use `pip-tools` to update dependencies.

Install `pip-tools` first:

```sh
pip install pip-tools
```

Compile `requirements.in` file to `requirements.txt` file:

```sh
pip-compile
```

Then synchronize dependencies with compiled `requirements.txt` files:

```sh
pip-sync
```

### Create input directories and datasets

Prepare directories for models savings, features savings and datasets:

```sh
python input.py
```

## Training

For LeNet5 network, running:

```sh
python train_lenet5.py
```

For VGG16 network, running:

```sh
python train_vgg16.py
```

Trained models are saved in `models/` directory.

## Serving HTTP server

Running an `aiohttp` HTTP server for uploadng image files for prediction:

```sh
python -m web.httpserver
```

The default listen port is 8080.
For see help for more options:

```sh
python -m web.httpserver --help
```

Access HTTP server in web brower in `http://server_address` and upload image files for prediction.

## References

Source code for traning(LeNet5 and VGG16 networks) is based on:

* https://mp.weixin.qq.com/s/2z3rAktRj-4pMmt5b8ASLA (in Chinese)
