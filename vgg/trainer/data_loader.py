#-*- coding: utf-8 -*-

import json
import numpy as np
from tensorflow.python.lib.io import file_io


def data_loader(filepath):

    with file_io.FileIO(filepath, "r") as f:
        data = json.load(f)

    images = np.array(data['images'])
    votes = np.array(data['votes'])

    return (images, votes)

