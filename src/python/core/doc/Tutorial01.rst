.. vim: set fileencoding=utf-8 :

======================================
 Tutorial 01. Initialization of Arrays
======================================

In this section we will illustrate the many different ways to initialize an array in |project|.
In summary you can

* specify shape or guess shape
* specify type or guess type

Simply allocate the space
-------------------------

Please look in :doc:`UsingBlitzArrays` for the different types of arrays.

We can create a simple array by directly specifying its shape.
This is equivalent to allocating (reserving) the memory.
Please not that there are no guaranties of the values. 

.. code-block:: python

   import torch

   # create a array of size 5 times 7, of type uint8
   my_array = torch.core.array.uint8_2(5, 7)

As the array is not guaranteed to be initialize during allocation,
it is often useful to set the whole array to either zeros or ones.

.. code-block:: python

   # set all the values (globally) to 1
   my_array.ones()

   # set all the values (globally) to 0
   my_array.zeros()

   # set all teh values to 17
   my_array.ones()
   my_array = my_array * 17

In the last line we take advantage of a useful feature in |project|.
It is possible to multiply all the elements in a array with a scalar as 
well as adding a scalar value to all of the elements

.. code-block:: python

   # multiply with 13
   my_array * 13

   # add 33
   my_array + 33

Direct initialization of array
------------------------------

We may initialize the elements directly when we create a array.
In the following example we create a 2D array of size 2x3 and specify the values to use for initialization.

.. code-block:: python

   # [1, 2, 3, 4, 5, 6] are the values and (2,3) is the 'shape'
   # the values will be fill row by row
   my_array = torch.core.array.float64_2([1, 2, 3, 4, 5, 6], (2,3))

Guess the type
--------------

Sometimes we do not want to care about the type to use.
A very useful feature is that you can use the core.array.array function (instead of 
example core.array.uint8_2).

.. code-block:: python

    # Please guess the type for me.
    my_array = torch.core.array.array([1, 2, 3, 4, 5, 6], (2,3))

Guess the shape
---------------

Construction of arrays from scratch requires always that you pass a
non-nested iterable followed by a shape. Sometimes you want python to
just do a best guess.

.. code-block:: python 

    # [[1, 2], [2, 3]] could be interpreted as a 2D integer array.
    t5_array_1 = torch.core.array.array([[1,2,3], [4,5,6]])
