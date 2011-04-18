.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue  5 Apr 07:46:12 2011 

====================
 Tutorial 01. Basic usage of Arrays
====================

In this section we will illustrate the basic python usage of Arrays

Some basic usage of Arrays / Matrices
-------------------------------------

.. code-block:: python

   import torch

   # create two Matrices
   A = torch.core.array.float64_2(3, 2)         # Straight forward
   B = torch.core.array.float64_2(A.shape())    # Using A' shape

   # Look closer
   # Notice that the elements in the matrices are not gauarantied to be initialized to zeros
   print A
   print B

   # Assign a value to "whole" Matrix
   A.ones()                    # Set all values to one
   B.zeros()                   # Set all values to zero

   # Look closer
   # The matrices are now properaly initialized
   print A
   print B

   # Mathematical operations
   print 0.45 * A             # It is possible to directly mulitple with a scalar
   print B + 5                # It is possible to add a scalar to all element in matrix

   print  0.45 * A + B + 1    # Example of rich expressions

Converting and casting in Python
--------------------------------

It is sometimes nessassary to cast or convert arrays.

.. code-block:: python

   import torch

