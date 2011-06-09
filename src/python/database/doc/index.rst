.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Sun  3 Apr 17:13:14 2011 

.. Index file for the Python Torch::database bindings

===================
 Data Input/Output
===================

Array
-----

The ``Array`` represents a ``blitz::Array<T,N>`` abstraction. It can hold
either a real ``blitz::Array<T,N>`` representation or be just a pointer to a
datafile where the actual array can be loaded from.

Arrayset
--------

The ``Arrayset`` represents a collection of ``torch.database.Array``'s.

HDF5File
--------

The ``HDF5File`` is our preferred container for all sorts of array and scalar
data to be produced or consumed from |project|.

Reference
---------

.. automodule:: torch.database
  :members:
