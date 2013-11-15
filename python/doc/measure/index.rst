.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Apr 20 08:19:36 2011 +0200
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

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
   f_score
   far_threshold
   farfrr
   frr_threshold
   min_hter_threshold
   min_weighted_error_rate_threshold
   mse
   ppndf
   precision_recall
   precision_recall_curve
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
   precision_recall_curve
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
