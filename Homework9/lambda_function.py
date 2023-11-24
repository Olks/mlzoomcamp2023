#!/usr/bin/env python
# coding: utf-8

import numpy as np

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request

from PIL import Image


model_file_name = "bees-wasps-v2.tflite"

interpreter = tflite.Interpreter(model_path=model_file_name)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


def predict(url):
    image = download_image(url)
    target_size = (150,150)
    image_resized = prepare_image(image, target_size)

    x = np.array(image_resized, dtype='float32')
    X = preprocess_input(x)

    interpreter.set_tensor(input_index, [X])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(result = float_predictions[0])


def lambda_handler(event, context):
    url = event['url']
    print(url)
    result = predict(url)
    print(result)
    return result

