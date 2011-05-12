.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue  5 Apr 07:46:12 2011 

===================================================
 Tutorial 01. Cropping face images using a database
===================================================

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


