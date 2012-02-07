.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <laurent.el-shafey@idiap.ch>
.. Tue Feb 6 23:08:30 2012 +0200
.. 
.. Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

=============
 SP Tutorial 
=============

Please keep in mind that numpy and scipy already provide significant features.
For this reason, we have not generated python bindings for some of the C++ functions
available in the C++ sp module of bob.
In the following we present a small example how to use the extrapolation function 
of the sp module. This might be useful before performing any convolution-like
operation.

.. code-block:: python

   import bob
   import numpy

   # create an psuedo image (instead of loading an image)
   image          = numpy.array([1,2,3,4,5], 'uint8')
   image_zero     = numpy.ndarray((15,), 'uint8')
   image_constant = numpy.ndarray((15,), 'uint8')
   image_nearest  = numpy.ndarray((15,), 'uint8')
   image_circ     = numpy.ndarray((15,), 'uint8')
   image_mirror   = numpy.ndarray((15,), 'uint8')
  
   bob.sp.extrapolateZero(image, image_circ)
   bob.sp.extrapolateConstant(image, image_constant, 37)
   bob.sp.extrapolateNearest(image, image_nearest)
   bob.sp.extrapolateCircular(image, image_circ)
   bob.sp.extrapolateMirror(image, image_mirror)
