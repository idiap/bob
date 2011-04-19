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

Now let's see a more comples ip fucntion: Face crop + normalization.

.. code-block:: python

   import torch

   # Because this operation is a bit more complicated than just cropping, we
   # need to create an object (instance of the FaceEyesNorm class).
   
   eye_distance = 33
   final_height = 80
   final_width  = 64
   overlap_h    = 0 # used if we need a bigger crop (height)
   overlap_w    = 0 # used if we need a bigger crop (width)
   my_face_normer = torch.ip.FaceEyesNorm(eye_distance, final_height, final_width, overlap_h, overlap_w) 

   # create an psuedo image (instead of loading an image)
   
   image = torch.core.array.uint8_2(240, 320)
   image.ones() 

   # as with all ip functions, we need to create the destination
   # our selfs

   dst = torch.core.array.uint8_2(final_height, final_width)

   # lets crop and normalize the image using eye locations
   # first we will start by specifiying the eye locations
   
   height_left_eye = 120
   width_left_eye  = 100
   
   height_right_eye = 130
   width_right_eye  = 140

   # we now crop and normalise by using the object (my_face_normer) we created above
   # not that it is smart to use the same object for many images, if all images
   # should be normalised and have the same final size.

   my_face_normer.__call__(image, dst, height_left_eye, width_left_eye, height_right_eye, width_right_eye)   


Part 2. Cropping face images with a database      
--------------------------------------------

This is a more extensive example how to crop images using a database

.. code-block:: rest
   <dataset>
     <pathlist>
       <entry path="/mnt/jupiter/databases_raw/BANCA_PGM_IMAGES/"/>
     </pathlist>
     <arrayset id="1" role="Pattern" elementtype="uint8" shape="576 720">
       <external-array id="1" codec="torch.image" file="9049_m_wm_s09_9049_en_4.pgm"/>
       <external-array id="2" codec="torch.image" file="1008_f_g1_s02_1010_en_5.pgm"/>
     </arrayset>
     <arrayset id="2" role="EyeCenters" elementtype="uint32" shape="4">
       <array id="1">
         197 319 195 385 
       </array>
       <array id="2">
         278 355 277 435 
       </array>
     </arrayset>
   </dataset>
   
.. code-block:: python

   import math
   import os, sys
   import unittest
   
   def width_to_eye_distance(width):
       # used to be the standard configuration in torch3/5
       return int(33./64. * width);
   
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
   
           return dst.cast('uint8')
