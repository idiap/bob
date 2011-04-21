.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue  5 Apr 07:46:12 2011 

===================================
 Tutorial 01. Basic usage of Arrays
===================================

In this section we will illustrate the basic python usage of Arrays.

Some basic usage of Arrays / Matrices
-------------------------------------

We use torch to create a simple array.

.. code-block:: python

   import torch

   # create a array of size 5 times 7, of type uint8
   my_array = torch.core.array.uint8_2(5, 7)

The array is not gauranted to be initialize during allocation.
It is therefore often useful to set the whole array to either zeros or ones.

.. code-block:: python

   # set all the values (globally) to 1
   my_array.ones()

   # set all the values (globally) to 0
   my_array.zeros()

   # set all teh values to 17
   my_array.ones()
   my_array = my_array * 17

In the last line we used a powerful feature in this library.
It is possible to multiply all the elements in a array with a scalar as 
well as adding a scalar value to all of the elements

.. code-block:: python

   # multiply with 13
   my_array * 13

   # add 33
   my_array + 33

If you wish to use a slice (part) of array this is possible with the normal 
python syntax

.. code-block:: python

   # select the first row
   my_array[1, :]

   # select the second column
   my_array[:, 2]


.. code-block:: python

   import torch

   # Create two 2D-double arrays  (float64_2) of size 5 times 7.
   # Terminology
   #   With "the type" we describe the number of dimensions (2D) and the data type (double = float64).
   #   With "the shape" we describe the size (5 times 7) of the array.

   A = torch.core.array.float64_2(5, 7)         # Specify directly the type and shape
   B = torch.core.array.float64_2(A.shape())    # Using A' shape

   # Closer look:  
   #    Notice that the elements in the matrices are not gauarantied to be initialized to zeros
   
   print A
   print B

   # It is possible to assign the whole matrix to either zeros or ones.

   A.ones()                    # Set all values to one
   B.zeros()                   # Set all values to zero

   # Look closer
   #    The arrays are now properaly initialized
   print A
   print B

   # |project| provide most mathematical operations in a powerful way

   print 0.45 * A             # It is possible to directly mulitple with a scalar
   print B + 5                # It is possible to add a scalar to all element in matrix

   print  0.45 * A + B + 1    # Example of rich expressions

Converting and casting in Python
--------------------------------

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
      
