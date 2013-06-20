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

.. Index file for the Python bob::math bindings

=============
 Mathematics
=============

This module binds math functions of |project| to python when there is no numpy
equivalent.

.. module:: bob.math

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   chi_square
   euclidean_distance
   histogram_intersection
   kullback_leibler
   linsolve
   linsolve_
   linsolve_cg_sympos
   linsolve_cg_sympos_
   linsolve_sympos
   linsolve_sympos_
   normalized_scalar_product
   norminv
   normsinv
   pavx
   pavx_
   pavxWidth
   pavxWidthHeight
   scatter
   scatter_

.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   LPInteriorPoint
   LPInteriorPointLongstep
   LPInteriorPointPredictorCorrector
   LPInteriorPointShortstep
