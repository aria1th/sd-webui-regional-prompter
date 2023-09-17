import base64
from io import BytesIO
import os
import gzip
import json
import re
import numpy as np
from PIL import Image

# Expects Image -> Binary -> (Gzip) -> Base64
def encode_image(image):
    """
    Standard expected input for Gradio image components.
    """
    if image is None:
        return None
    return base64.b64encode(gzip.compress(image)).decode('ascii')

def decode_data(data:str):
    """
    Standard expected output for Gradio image components.
    """
    # check if data is actual path
    if os.path.isfile(data) and len(data) < 256:
        # if json, then load json and create as numpy array (mask)
        if data.endswith(".json"):
            raise NotImplementedError("JSON loading not implemented yet")
        elif data.endswith(".npy"):
            return np.load(data)
        else:
            # expects image
            image = Image.open(data)
            if image.mode == "RGBA":
                # convert to RGB
                image = image.convert("RGB")
            return np.array(image)
    # check if data is base64 encoded
    if re.match(r"^[A-Za-z0-9+/]+[=]{0,2}$", data):
        # try to decode
        try:
            bytes_data = base64.b64decode(data)
        except:
            raise ValueError("Could not decode as base64 data")
    else:
        # find invalid characters
        invalid_chars = re.findall(r"[^A-Za-z0-9+/=]", data)
        if invalid_chars:
            raise ValueError("Invalid characters found in data: {}".format(invalid_chars))
    assert isinstance(bytes_data, bytes), "Expected bytes data, got {}".format(type(bytes_data))
    # check if data is gzip compressed
    if is_compressed(bytes_data):
        # try to decompress
        try:
            bytes_data = gzip.decompress(bytes_data)
        except:
            raise ValueError("Could not decompress as gzip data")
    # try to load as numpy array
    try:
        bytesIO = BytesIO(bytes_data)
        image_arr = np.array(Image.open(bytesIO))
        image = Image.fromarray(image_arr)
        # remove alpha channel and convert to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return np.array(image)
    except:
        raise ValueError("Could not load as image data")

def is_compressed(data:bytes)->bool:
    """
    Check if data is gzip compressed.
    """
    return data[:2] == b'\x1f\x8b'


