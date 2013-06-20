.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Sun Apr 3 19:18:37 2011 +0200
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

.. Index file for the Python bob::sp bindings

===================
 Signal Processing
===================

Signal processing utilities.

.. module:: bob.sp

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   dct
   extrapolate
   extrapolate_circular
   extrapolate_constant
   extrapolate_mirror
   extrapolate_nearest
   extrapolate_zero
   fft
   fftshift
   idct
   ifft
   ifftshift

.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   BorderType
   DCT1D
   DCT1DAbstract
   DCT2D
   DCT2DAbstract
   FFT1D
   FFT1DAbstract
   FFT2D
   FFT2DAbstract
   IDCT1D
   IDCT2D
   IFFT1D
   IFFT2D
   Quantization
   SizeOption
   quantization_type
