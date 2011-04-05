.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue  5 Apr 07:46:12 2011 

====================
 Using Blitz Arrays
====================

|project| provides a python *bridge* to `Blitz`_ arrays. The intent of this
bridge is not to create an exhaustive wrapping, but to allow simple to
mid-complexity scripts to run without requiring the user to recur to `NumPy`_
arrays and avoid data copying whenever possible. The |project|-Blitz array
python bridge supports the ``__array__()`` python protocol making it
possible to write very complex programs that use both `NumPy`_ (or any other
array types) together with Blitz arrays. The array protocol also makes it
possible to pass Blitz arrays to third-party libraries that require an array
interface (e.g. `Matplotlib`).

This section introduces most of the implemented bridge functionality.
Extensive coverage is available at the reference manual at the end

Supported Types
---------------

Because Blitz arrays are template types, we must fix the set of supported
combinations of element type and number of dimensions that are supported in the
bridge. Please refer to :doc:`../../../cxx/core/doc/UsingBlitzArrays` for a
detailed list and element names.

.. note::
  
  We use the same element type names as those defined for `NumPy`.

Reference Manual
----------------

.. note::

  In this reference guide, please consider ``type_D`` (e.g. ``bool_1``) the
  same as ``blitz::Array<type,D>`` (e.g. ``blitz::Array<bool,1>``).

.. automodule:: torch.core.array
  :members:
  
.. Place here your references
.. _blitz: http://www.oonumerics.org/blitz
.. _numpy: http://http://numpy.scipy.org
