.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Apr 20 08:19:36 2011 +0200
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

.. Index file for the Python bob::measure bindings

=========
 Metrics
=========

.. module:: bob.measure

.. rubric:: Module utilities

.. autosummary::
   :toctree: generated/

   cmc
   correctly_classified_negatives
   correctly_classified_positives
   det
   eer_rocch
   eer_threshold
   epc
   far_threshold
   farfrr
   frr_threshold
   min_hter_threshold
   min_weighted_error_rate_threshold
   mse
   ppndf
   recognition_rate
   relevance
   rmse
   roc
   roc_for_far
   rocch
   rocch2eer

.. module:: bob.measure.plot

.. rubric:: Plotting

.. autosummary::
   :toctree: generated/

   cmc
   det
   det_axis
   epc
   roc

.. module:: bob.measure.load

.. rubric:: File Parsing and Loading

.. autosummary::
   :toctree: generated/

   cmc_five_column
   cmc_four_column
   five_column
   four_column
   split_five_column
   split_four_column
