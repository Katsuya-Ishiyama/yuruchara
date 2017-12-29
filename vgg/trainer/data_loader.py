#-*- coding: utf-8 -*-

import json
import numpy as np


def data_loader(filepath):

    with open(filepath) as f:
        data = json.load(f)

    images = np.array(data['images'])
    votes = np.array(data['votes'])

    return (images, votes)

