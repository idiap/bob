import math
import os, sys
import unittest
import torch

def width_to_eye_distance(width):
    # used to be the standard configuration in torch3/5
    return int(33./64. * width);

def height_offset(crop_height):
    return int(1. / 3. * crop_height)

def width_offset(crop_width):
    return int(0.5 * crop_width)

class Cropper():
    def __init__(self, xml_file, crop_height = 80, crop_width = 64):
        self.db  = torch.io.Dataset(xml_file)

        # cropping parameters
        self.H  = crop_height
        self.W  = crop_width
        self.ED = width_to_eye_distance(self.W)

        # we need to specify the center between the eyes
        self.OH = height_offset(self.H)
        self.OW = width_offset(self.W)

        self.IMAGE_SET_INDEX      = 1
        self.EYECENTERS_SET_INDEX = 2

        # WARNING, before the api demanded two more numbers (0, 0)
        self.GN = torch.ip.FaceEyesNorm(self.ED, self.H, self.W, self.OH, self.OW)

    def size(self):
        """ Return the size of the array, this is not very stabile """
        return self.db.arraysets()[0].__len__()

    def new_dst(self):
        # the dst shape is stolen from the cxx file.
        return torch.core.array.float64_2(self.H, self.W)

    def index(self, index):
        """ Extract only one image (cropped/normalized) from the dataset """

        # extract the RGB/gray image and the eye-center coordinates
        tmp_img = self.db[self.IMAGE_SET_INDEX     ][index].get()
        crd     = self.db[self.EYECENTERS_SET_INDEX][index].get()

        # turn the RGB image to gray if needed
        global img
        if 3 == tmp_img.dimensions():
            img = torch.ip.rgb_to_gray(tmp_img)
        else:
            img = tmp_img

        # cropp coordinates
        LH = int(crd[0]); LW = int(crd[1]); RH = int(crd[2]); RW = int(crd[3])

        # create a destination array
        dst = self.new_dst()

        # do the actual cropping
        self.GN.__call__(img, dst, LH, LW, RH, RW)

        # cast and return the image
        return dst.cast('uint8')

    def get_all(self):
        """ Get all the cropped/normalized images """
        crops = []
        for iii in range(1, self.size() + 1):
            crops.append(self.index(iii))
        return crops

