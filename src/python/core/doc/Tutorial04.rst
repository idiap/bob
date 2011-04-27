====================================
 Tutorial 04. Interacting with numpy
====================================

In this example we will illustrate some of the ways to use numpy and torch together

.. code-block:: python

  import numpy
  import torch

  # create some data using torch
  data = torch.core.array.array([[1, 2], [1, 3], [1, 2], [1, 3], [1, 2]])

  # perform SVD using numpy directly on the torch array
  s,v,d = numpy.linalg.svd(data)

  # turn the answer from the numpy operation (SVD) back to torch array format
  s = torch.core.array.array(s)
  v = torch.core.array.array(v)
  d = torch.core.array.array(d)
