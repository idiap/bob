==============================================
 Tutorial 03. Converting and casting in Python
==============================================

There are many cases where we have to cast or convert arrays.
When casting it is necessary to pass a string with the new type.
The possible types are listed here: TODO.

.. code-block:: python

   import torch

   # We will illustrate a simple cast between uint8 and float64

   # Create a 2D uint8 array of size 4 times 5.
   # Make sure it is properly initialized (set it to ones)
   # Look closer (by printing the data).

   A = torch.core.array.uint8_2(4,5)
   A.ones()
   print A

   # All elements are exactly 1. 
   #
   # [[1 1 1 1 1]
   #  [1 1 1 1 1]
   #  [1 1 1 1 1]
   #  [1 1 1 1 1]]
   #

   # Now cast the array to double (float64)

   B = A.cast("float64")
   print B

   # All the elements are now 1. (notice the dot).
   #
   # [[ 1.  1.  1.  1.  1.]
   #  [ 1.  1.  1.  1.  1.]
   #  [ 1.  1.  1.  1.  1.]
   #  [ 1.  1.  1.  1.  1.]]
   #   

Sometimes it is better to convert the array instead of simply cast it. 
We we present a couple of examples below.

.. code-block:: python



   import torch

   A = torch.core.array.uint8_2(4,5)
   A.ones()
