===================================
 Tutorial 02. Basic usage of Arrays
===================================

If you wish to use a slice (part) of array this is possible with the normal 
python syntax.

.. code-block:: python

   import torch

   # create an array (guess size and shape)
   my_array = torch.core.array.array([[1,2,3], [4,5,6]])

   # select the first row
   # Please note that the answer is a 1D array with the first dimension containing a single row
   my_array[0, :]

   # select the second column
   my_array[:, 1]

Some mathematical operations on arrays.

.. code-block:: python

   import torch

   # create two arrays (guess type and shape)
   A = torch.core.array.array([[1,2,3], [4,5,6]])
   B = torch.core.array.array([[7,7,8], [8,8,9]])

   # It is possible to directly multiple with a scalar
   A1 = 0.45 * A

   # It is possible to add a scalar to all element in matrix
   B1 = B + 5

   # Example of rich expressions
   C = 0.45 * A + B / 0.18 + 1

For a complete list of possible mathematical expressions look here : TODO
