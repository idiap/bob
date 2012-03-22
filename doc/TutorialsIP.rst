.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
.. 
.. Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

.. testsetup:: *
  
  import bob
  import numpy
  import math
  import os

  image_path = os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'cxx/ip/test/data/image_r10.pgm')
  color_image_path = os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'cxx/ip/test/data/imageColor.ppm')


*****************************
 Image and signal processing
*****************************

Introduction
============

As already mentioned in the Array section, signals and images are prepresented as `NumPy`_ **ndarray**'s. In the previous section we have already learned, how the data can be read and written using the **bob.io** API. This section will give a deeper insight in some simple and some more complex image and signal processing utilities of **bob**.


Simple image processing
=======================

The basic operations on images are the affine image conversions like image scaling, rotation, and cutting. 


Scaling images
~~~~~~~~~~~~~~

To compute a scaled version of the image, simply create the image in the desired scale:

.. doctest::
  :options: +NORMALIZE_WHITESPACE
 
  >>> A = numpy.array( [ [1, 2, 3], [4, 5, 6] ], dtype = numpy.uint8 ) # A small image of size 2x3
  >>> print A
  [[1 2 3]
   [4 5 6]]
  >>> B = numpy.ndarray( (3, 5), dtype = numpy.float64 )               # A larger image of size 3x5

and call the scale function of **bob**: 

  >>> bob.ip.scale( A, B )
  >>> print B
  [[ 1.   1.5  2.   2.5  3. ]
   [ 2.5  3.   3.5  4.   4.5]
   [ 4.   4.5  5.   5.5  6. ]]
  
which bi-linearly interpolates image A to image B. Of course, scaling factors can be different in horizontal and vertical direction:

  >>> C = numpy.ndarray( (2, 5), dtype = numpy.float64 )
  >>> bob.ip.scale( A, C )
  >>> print C
  [[ 1.   1.5  2.   2.5  3. ]
   [ 4.   4.5  5.   5.5  6. ]]



Rotating images
~~~~~~~~~~~~~~~

The rotation of an image is slightly more difficult since the resulting image size has to be computed in advance. For that purpose, the **bob.ip.get_rotated_image_size** function can be used:

  >>> A = numpy.array( [ [1, 2, 3], [4, 5, 6] ], dtype = 'uint8' ) # A small image of size 3x3
  >>> print A
  [[1 2 3]
   [4 5 6]]
  >>> rotated_shape = bob.ip.get_rotated_output_shape( A, 90 )
  >>> print rotated_shape
  (3, 2)
   
After the creation of the rotated version of the image, the **bob.ip.scale** function can be used:
  
  >>> A_rotated = numpy.ndarray( rotated_shape, dtype = 'float64' ) # A small image of rotated size
  >>> bob.ip.rotate(A, A_rotated, 90)      # execute the rotation
  >>> print A_rotated
  [[ 3.  6.]
   [ 2.  5.]
   [ 1.  4.]]



Color type conversion
~~~~~~~~~~~~~~~~~~~~~

When dealing with color images, sometimes different parts of the color image are required. Most common face verification algorithms require the images to be gray scale. To assure that the image that is loaded is actually a gray level image, the conversion from color to gray scale images can be applied:

  >>> # set up 'color_image_path' to point to any kind of image
  >>> image = bob.io.load( color_image_path )
  >>> if image.ndim == 3:                    # Test if the loaded image is a color image
  ...   gray_image = numpy.ndarray( image.shape[1:3], dtype = image.dtype )  # create gray image in desired dimensions
  ...   bob.ip.rgb_to_gray( image, gray_image )                              # Convert it to gray scale
  ...   image = gray_image

Converting a colored RGB image to YUV is as straightforward:

  >>> rgb_image = bob.io.load( color_image_path )
  >>> yuv_image = numpy.ndarray( rgb_image.shape, dtype = rgb_image.dtype )
  >>> bob.ip.rgb_to_yuv( rgb_image, yuv_image )



Complex image operations
========================

Complex image operations are usually wrapped by classes. The usual workflow is to first generate an object of the desired class, specifying parameters that are independent on the images to operate, and to second use the class on images. Usually, objects that perform image operations have the __call__ function overloaded, so that one simply can use it as if it were functions.

Image filtering
~~~~~~~~~~~~~~~

One simple example of image filtering is to apply a Gaussian blur filter to an image. This can be easily done by first creating an object of the bob.ip.Gaussian class:

  >>> filter = bob.ip.Gaussian( radius_y = 1, radius_x = 1, sigma_y = 0.3, sigma_x = 0.3)
  
Now, let's see what happens to a small test image:

  >>> test_image = numpy.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]], dtype='float64')
  >>> filtered_image = numpy.ndarray(test_image.shape, dtype='float64')
  >>> filter(test_image, filtered_image)
  >>> print filtered_image
  [[ 0.93562108  0.06327015  0.00221754  0.06327015  0.93562108]
   [ 0.06327015  0.87345971  0.09324206  0.87345971  0.06327015]
   [ 0.00221754  0.09324206  0.87567725  0.09324206  0.00221754]
   [ 0.06327015  0.87345971  0.09324206  0.87345971  0.06327015]
   [ 0.93562108  0.06327015  0.00221754  0.06327015  0.93562108]]


See, we ended up with a nicely smoothed cross.


Another filter you might want to us are Gabor filters.

..   >>> complex_image = image.astype(complex)
..   >>> filtered_image = numpy.ndarray(complex_image.shape, dtype = 'complex')
..   >>> kernel = bob.ip.GaborKernel(complex_image.shape, (0,1))
..   >>> kernel(complex_image, filtered_image)
..   >>> abs_image = numpy.abs(filtered_image)
..   >>> bob.io.Array(abs_image).save("/scratch/mguenther/test.hdf5")


Normalizing images according to eye positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many biometric applications to faces, the images are geometrically normalized according to the eye positions, which are either hand-labeled or detected by an algorithm. The first thing to do is to create an object of the class, defining the image properties of the geometrically normalized image (that will be generated when applying the object):

  >>> face_eyes_norm = bob.ip.FaceEyesNorm(eyes_distance = 64, crop_height = 128, crop_width = 128, crop_eyecenter_offset_h = 32, crop_eyecenter_offset_w = 64)

Now, we have set up our object to generate images of size (128, 128) that will put the left eye to pixel position (32, 32) and the right eye to position (32, 96). Afterwards, this object is used to geometrically normalize the face, given the eye positions in the original face image:

  >>> face_image = bob.io.load( image_path )
  >>> cropped_image = numpy.ndarray( (128, 128), dtype = 'float64' )
  >>> face_eyes_norm( face_image, cropped_image, le_y = 67, le_x = 47, re_y = 62, re_x = 71)





Signal Processing
=================

* Image rescaling, rotating, etc.

* Image filter (LBP, Gabor, etc.)

.. Place here your external references

.. _numpy: http://numpy.scipy.org

