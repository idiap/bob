==============================================
Practical torch5spro: Face verification system
==============================================

Imagine the following setup: you want to create a face verification system for yourself.
You have 3 images of yourself (frontal faces) that you want to use as reference.
The goal is that only you (your face) should unlock a system and that everyone else should be considered to be impostors.

+---------------------------------------+----------------------------------------+----------------------------------------+
|.. image:: 1001_f_g1_s01_1001_en_1.jpg | .. image:: 1001_f_g1_s01_1001_en_2.jpg | .. image:: 1001_f_g1_s01_1001_en_3.jpg |
|   :height: 144                        |    :height: 144                        |    :height: 144                        |
|   :width: 180                         |    :width: 180                         |    :width: 180                         |
|   :alt: Reference 1                   |    :alt: Reference 2                   |    :alt: Reference 3                   |
+---------------------------------------+----------------------------------------+----------------------------------------+

(We will refer to these 3 images as training images.)

The problem of using the 3 images directly for template matching (when you match a new image pixel-by-pixel to an old) 
is that the "space" is too large.
Even if the images are relatively small, say 80 by 64, the possible variations of an image are 64*80*256*3 (height * width * pixel value * 3 colors).
To compare two vectors (images) in this 64*80*256*3 dimensional space is not optimal and will yield bad results.
We would therefore like to understand/learn a subspace in which it is easier. 
This approach to the problem was one of the first successful techniques in face recognition an is refered to as Principal Component Analysis (PCA).
PCA is a technique that finds the optimal subspace (principal directions) to represent our images.
Because the image-form of these principal components look like "faces" and that PCA is a form of eigenvalue decomposition, this techniques is often refered to as Eigenface composition.

In this practical tutorial we will demonstrate one possible way to implement a eigenface face verifier in torch5spro.

In summary we will perform the following steps to train our system

* Derive a subspace which represent a "face"
* Using the above subspace, create a model of the user
* Compare an unknown image to the model of the user


Deriving a better representation (finding principal components)
---------------------------------------------------------------

.. image:: dia-1.png

Imagine that you toke all the photos of all your friends.
You crop those images so only the face is visible and you align their eye-centers.
If you take the average of all this images, you will get an average "face" of all your friends.

If you now consider the difference between each individual face (friend) compared to this average face,
you can understand that the difference is not completely arbitrary.
There is some sort of manifold/subspace/representation in which "all" faces lie.
It is this space we are searching for.

When implementing our system in torch5spro we approach the problem very much the same.
We start by collecting as many photos as we can from any different individuals.
This collection is refered to as the world set.
We crop, normalize and align all those faces.
Thereafter we compute the average face.
The task of cropping a set of images using a database we have covered in this practical example
:ref:`practical-cropping-images`.
  
(Please note that the difference between an image and an vector is just the representation.
An image is a 2D array while an vector is a 1D array. 
If we pick out the values row-by-row from the image (2D array) we can easily create a 1D array (size: rows * cols).)

.. code-block:: python

  import Cropper

  # crop all the images in our database
  myCropper = Cropper.Cropper('my-world-database.xml')
  world_images = myCropper.get_all()

  # turn all images into vectors (from 2D arrays -> 1D arrays)
  # the function vectorOf is a help function to exactly this
  world_vectors = map(torch.core.array.uint8_2.vectorOf, world_images)


Create a model of the user
--------------------------

.. image:: dia-2.png


Test system with unknown image
------------------------------

.. image:: dia-3.png





