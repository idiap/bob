========================================
 Tutorial 05. Some useful help functions
========================================

In order to make the ip (image processing) part of |project| fast, the user is 
responsible to allocate the destination of the operation. For example, if you 
wish to run a Gaussian filter over an image (2d array) you have to allocate a 
array (2d) of the same size and shape as the original image.

To make this process a bit easier there are three help functions defined in |project|:

* sameAs()
* grayAs()
* asOneRow()

sameAs() - to allocate an array of the same size and shape
----------------------------------------------------------

sameAs() - to allocate an array of the same size and shape without copying the data.
This is useful when working with the ip package. An example use is the following,

.. code-block:: python

  # create an array (pseudo image)
  src_image = torch.core.array.uint8_2([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], (4, 4))

  # allocate a destination
  dst_image = src_image.sameAs()

  # Smooth the image (here we are using a 3x3 Gaussian filter)
  # MySmoother = torch.ip.Gaussian(3, 3, 0.25)
  # MySmoother(src_image, dst_image)
  # TODO, It seems like we have to have a double as output :(

grayAs() - when converting from color images to gray
----------------------------------------------------

When converting a color image to a gray-scale image we need to have allocate an array
of the same size but that is 2D instead of 3D. To ease the burden there is a function called
grayAs() that will do exactly that. 

.. code-block:: python

  # imagine a color image (pseudo)
  src_image = torch.core.array.uint8_3([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (3, 2, 2))

  # a gray scale version of the src image. same size and shape but without
  # copying the data
  dst_image = src_image.grayAs()

  # transform the pseudo RGB image to gray-scale
  torch.ip.rgb_to_gray(src_img, dst_img)

asOneRow() - to take an 2D array and turn it into 1D
----------------------------------------------------

It is often necessary to transform a 2D array into an 1D array. 
A good example is when you need to create a sample out of an image.
To do this there is a help function implemented which is called asOneRow().
This function takes each row in the 2D array and saves it into one big array.

.. code-block:: python

  # create an array (pseudo image)
  src_image = torch.core.array.uint8_2([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], (4, 4))

  # turn into a 1D vector
  my_sample = src_image.asOneRow()




  


