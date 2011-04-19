.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue  5 Apr 07:46:12 2011 

====================
 Tutorial 01. Cropping face images using a database
====================

Part 1. Basic usage of an ip functionality
------------------------------------------

In this section we present a small example how to use a ip (image processing) function.
Our goal is to take an image, in our case a 2D uint8 array, and crop it.

.. code-block:: python

   import torch

   # create an psuedo image (instead of loading an image)
   image = torch.core.array.uint8_2(80, 64)
   image.ones() 
   
   # Some of the ip functionality is a simple function call
   # Whereas some ip functionality are more complex and you
   # have to create a class object. 
   # Whichever case you always have to create, yourself,
   # the destination of the operation.

   # example of destination
   crop_height = 40
   crop_width  = 32
   my_crop = torch.core.array.uint8_2(crop_height, crop_width)

   # crop the image and store in my_crop (which acts as our destination)
   top_left_height  = 0
   top_left_width   = 0
   torch.ip.crop(image, my_crop, top_left_height, top_left_width, crop_height, crop_width)


Part 2. Cropping face images with a database      
--------------------------------------------

.. code-block:: python

   import math
   import os, sys
   import unittest
   
   def width_to_eye_distance(width):
       # used to be the standard configuration in torch3/5
       return int(33./64. * width);
       # return int(40./64. * width);
   
   def height_offset(crop_height):
       return int(1. / 3. * crop_height)
   
   def width_offset(crop_width):
       return int(0.5 * crop_width)
   
   class Cropper():
       def __init__(self, xml_file):
           self.xml = xml_file
   
           self.db  = torch.database.Dataset(xml_file)
   
           # cropping parameters
           self.H  = 80
           self.W  = 64
           self.ED = width_to_eye_distance(self.W)
   
           # we need to specify the center between the eyes
           self.OH = height_offset(80)
           self.OW = width_offset(64)
   
           self.IMAGE_AS_INDEX      = 1
           self.EYECENTERS_AS_INDEX = 2
   
           # WARNING, before the api demanded two more numbers (0, 0)
           self.GN = torch.ip.FaceEyesNorm(self.ED, self.H, self.W, self.OH, self.OW) 
   
       def size(self):
           return min(len(torch.database.arrayset_array_index(self.db[self.IMAGE_AS_INDEX])),
                      len(torch.database.arrayset_array_index(self.db[self.EYECENTERS_AS_INDEX])))
   
       def new_dst(self):
           # the dst shape is stolen from the cxx file.
           return torch.core.array.float64_2(self.H, self.W)
   
       def get_DB(self):
           return self.db
   
       def index(self, index):
           img = self.db[self.IMAGE_AS_INDEX][index].get()
           crd = self.db[self.EYECENTERS_AS_INDEX][index].get()
   
           # cropp coordinates
           LH = int(crd[0])
           LW = int(crd[1])
           RH = int(crd[2])
           RW = int(crd[3])
   
           # 
           dst = self.new_dst()
   
           # do the actual cropping
           self.GN.__call__(img, dst, LH, LW, RH, RW)
   
           return dst.cast('uint8') # WARNING int8 does not work. Try convert(dst, 0, 255, 0.0, 1.0)
