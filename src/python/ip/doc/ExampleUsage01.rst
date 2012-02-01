.. vim: set fileencoding=utf-8 :
.. Niklas Johansson <niklas.johansson@idiap.ch>
.. Tue Apr 19 08:48:57 2011 +0200
.. 
.. Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

====================================
 Tutorial 01. Basic ip functionality
====================================

In this section we present a small example how to use a ip (image processing) function.
Our goal is to take an image, in our case a 2D uint8 array, and crop it.

.. code-block:: python

   import bob
   import numpy

   # create an psuedo image (instead of loading an image)
   image = numpy.ones((80, 64), 'uint8')
   
   # Some of the ip functionality is a simple function call
   # Whereas some ip functionality are more complex and you
   # have to create a class object. 
   # Whichever case you always have to create, yourself,
   # the destination of the operation.

   # example of destination
   crop_height = 40
   crop_width  = 32
   my_crop = numpy.ndarray((crop_height, crop_width), 'uint8')

   # crop the image and store in my_crop (which acts as our destination)
   top_left_height  = 0
   top_left_width   = 0
   bob.ip.crop(image, my_crop, top_left_height, top_left_width, crop_height, crop_width)

Now let's see a more complete ip function: Face crop + normalization.

.. code-block:: python

   import bob

   # Because this operation is a bit more complicated than just cropping, we
   # need to create an object (instance of the FaceEyesNorm class).
   
   eye_distance = 33
   final_height = 80
   final_width  = 64
   overlap_h    = 0 # used if we need a bigger crop (height)
   overlap_w    = 0 # used if we need a bigger crop (width)
   my_face_normer = bob.ip.FaceEyesNorm(eye_distance, final_height, final_width, overlap_h, overlap_w) 

   # create an pseudo image (instead of loading an image)
   
   image = numpy.ones((240, 320), 'uint8')

   # as with all ip functions, we need to create the destination
   # our selves

   dst = numpy.ndarray((final_height, final_width), 'uint8')

   # lets crop and normalize the image using eye locations
   # first we will start by specifying the eye locations
   
   height_left_eye = 120
   width_left_eye  = 100
   
   height_right_eye = 130
   width_right_eye  = 140

   # we now crop and normalize by using the object (my_face_normer) we created above
   # not that it is smart to use the same object for many images, if all images
   # should be normalized and have the same final size.

   my_face_normer.__call__(image, dst, height_left_eye, width_left_eye, height_right_eye, width_right_eye)   


