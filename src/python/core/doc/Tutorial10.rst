=================================
 Tutorial 10. Eigenfaces in torch
=================================

In this practical example we will do a eigenface classifier only using core functionality of torch.
Later we will show something similar but with more powerful torch tools. (TODO).

We will start by only importing the necessary modules.

.. code-block:: python

  # import all important
  import numpy
  import scipy
  import torch

Start by reading in all the filenames of images that we will use.

.. code-block:: python
  
  # read in all the image filenames
  f = open('images.lst','r')
  images=list()
  for line in f.readlines():
      images.append(line.rstrip('\n'))
  
  f.close()
  
Read up all the images and turn them into vectors.
Thereafter, concatenate all the vectors to one single big matrix.

.. code-block:: python

  # make all the images into one matrix
  
  # make the first image the base
  # signal that we use the global data variable

  image = images.pop()
  tmp = scipy.misc.imread(image) ## TODO !!

  global data
  data = tmp.reshape(tmp.size)
  accepted_size=data.size
  
  # rest
  for image in images:
  
      # print the filename
      # print "image filename: ", image
  
      # read in image (numeric image, make sure it is gray !:("
      tmp = scipy.misc.imread(image)
  
      # make it into one vector
      tmp = tmp.reshape(tmp.size)
  
      # concatenate
      if tmp.size == accepted_size:
          # print "good size: ", tmp.size
          data = scipy.vstack((data, tmp))
      #else:
      #    print "skipping vector of size: ", tmp.size
  
Perform the eigenface decomposition using SVD (PCA)

.. code-block:: python

  #
  number_of_samples = scipy.size(data, 0)
  dims_of_data      = scipy.size(data, 1)
  
  print ""
  print "SHOULD WE DO PCA?"
  if number_of_samples < dims_of_data:
      print "NO:( - not good", number_of_samples, " < ", dims_of_data
      print ""
  else:
      print "Yes"
      global U, S, V
      U, S, Vh = numpy.linalg.svd(data); V=Vh.T
  
  
  # columns of V are eigenvectors, to the values in S
   # we have to transpose the matrix
  VT = V.transpose()
  
  # save all the image
  eigen_faces = scipy.size(VT, 0)
  for face_nb in range (0, eigen_faces):
      scipy.misc.imsave( "eigen_face_%d.jpg" % (face_nb), VT[face_nb].reshape(47, 33))
