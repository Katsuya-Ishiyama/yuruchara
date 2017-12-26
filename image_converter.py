# -*- coding: utf-8 -*-

import csv
import json
import os
import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import resize


WORK_DIR = os.getcwd()

CROP_POSITION_LEFT = 0
CROP_POSITION_UPPER = 0
CROP_POSITION_RIGHT = 650
CROP_POSITION_BOTTOM = 650

RESIZE_WIDTH = 225
RESIZE_HEIGHT = 225

ENCODING = 'UTF-8'


def get_src_image_path(character_type, filename):

    """ Getting image filepath to be converted. """

    return os.path.join(WORK_DIR, character_type, 'src', 'image', filename)


def get_src_data_path(character_type):

    """ Getting data file path of including numbers of votes """

    return os.path.join(WORK_DIR, character_type, 'yuruchara.csv')


def get_output_data_path(character_type):

    """ Getting file path of resulting data """

    return os.path.join(WORK_DIR, character_type, '{}.json'.format(character_type))


def check_gif(filename):

    """ Checking whether GIF image. """

    extension = os.path.splitext(filename)[1]
    return (extension == '.gif')


def make_json(character_type, data_num=None):

    """ Converting data

    This function makes images and corresponding numbers of votes
    into 1 file as json.

    Args:
        character_type (str) : types of votes ranking.
        data_num       (int) : number of images which will be converted.
                               typically, to for testing learning.

    Returns:
        json data.
    """

    train_image_list = []
    point_list = []

    with open(get_src_data_path(character_type), 'r', encoding=ENCODING) as f:
        reader = csv.DictReader(f)

        for record in reader:

            if data_num is not None:
                read_line_num_exclude_header = reader.line_num - 1
                if read_line_num_exclude_header > data_num:
                    break

            # gif画像はresizeできないのでスキップする
            if check_gif(record['filename']):
                continue

            src_image = imread(get_src_image_path(character_type, record['filename']))
            cropped_image = src_image[CROP_POSITION_LEFT:CROP_POSITION_RIGHT, CROP_POSITION_UPPER:CROP_POSITION_BOTTOM, :]
            training_image = resize(image=cropped_image, output_shape=(RESIZE_HEIGHT, RESIZE_WIDTH))
            train_image_list.append(training_image.tolist())
            point_list.append(record['point'])

    return {'votes': point_list, 'images': train_image_list}


if __name__ == '__main__':

    for character_type in ['company', 'gotochi']:
        with open(get_output_data_path(character_type), 'w', encoding=ENCODING) as f:
            json.dump(make_json(character_type, data_num=3), f)

