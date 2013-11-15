.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Sun Apr 3 19:18:37 2011 +0200
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

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
   histogram_intersection
   kullback_leibler
   linsolve
   linsolve_
   linsolve_cg_sympos
   linsolve_cg_sympos_
   linsolve_sympos
   linsolve_sympos_
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
