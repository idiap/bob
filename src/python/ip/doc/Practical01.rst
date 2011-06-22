.. _practical-cropping-images:

================================================================
 Practical |project|: Cropping a set of images using a dataset
================================================================

.. warning::

  This section is outdated. It still refers to the obsolete database format.
  Please fix it ASAP.

It is common to have a large set of images that you would like to process in
the same way. In this section we will take a set of images (three) and crop
and normalise them using the |project|'s database format.

Below is an example of the |project| database syntax.

.. code-block:: xml
  
   <dataset>

     <!-- A relative or absolut path to prefix the pathnames -->
     <pathlist>
       <entry path="data/"/>
     </pathlist>

     <!-- The three images in our database of size 576 times 720-->
     <arrayset id="1" role="Pattern" elementtype="uint8" shape="576 720">
       <external-array id="1" codec="torch.image" file="1001_f_g1_s01_1001_en_1.jpg"/>
       <external-array id="2" codec="torch.image" file="1001_f_g1_s01_1001_en_2.jpg"/>
       <external-array id="3" codec="torch.image" file="1001_f_g1_s01_1001_en_3.jpg"/>
     </arrayset>

     <!-- Eye-center coordinates corresponding to the thee images above -->
     <arrayset id="2" role="EyeCenters" elementtype="uint32" shape="4">
       <!-- Please note that height comes before width -->
       <array id="1">
         286 383 288 468
       </array>
       <array id="2">
         307 386 310 474 
       </array>
       <array id="2">
         287 396 288 483 
       </array>
     </arrayset>

   </dataset>


Below we give an example of one way to put it all together. 
There are a couple of parameters that are hard coded (e.g, ratio of face).
However, we want to give a larger example how to but everything together.

.. code-block:: python

  import math
  import os, sys
  import unittest

  import torch

  def width_to_eye_distance(width):
      """ A function to calculate the eye-distance given a width of the face """
      return int(33./64. * width)
  
  def height_offset(crop_height):
      """ A function to calculate the location (height) of the face in the crop """
      return int(1. / 3. * crop_height)
  
  def width_offset(crop_width):
      """ For completeness, calculate the location (width) of the face in the crop """
      return int(0.5 * crop_width) # simply put face in the middle
  
  class Cropper():
      """ 
      This calls tries to wrapp the job of reading up the images and cropping information.
      In the end we simple want to get all images, cropped, normalized and nice.
      """      

      def __init__(self, xml_file, crop_height = 80, crop_width = 64):
          self.db  = torch.io.Dataset(xml_file)
  
          # cropping parameters
          self.H  = crop_height
          self.W  = crop_width
          self.ED = width_to_eye_distance(self.W)
  
          # we need to specify the center between the eyes
          self.OH = height_offset(self.H)
          self.OW = width_offset (self.W)
  
          # here we hard code the index of the database
	  # which arrayset holds the images and which holds the cropping information
          self.IMAGE_SET_INDEX      = 1
          self.EYECENTERS_SET_INDEX = 2
  
          # The actual instance of the "Face normalizer"
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
              img = tmp_img.grayAs()
              torch.ip.rgb_to_gray(tmp_img, img)
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

Using the simple class above, it is fairly easy to crop a large number of images.

.. code-block:: python

  import Cropper
  
  C = Cropper.Cropper('my-database-file.xml')
  images = C.get_all()
