.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Sun Apr 3 19:18:37 2011 +0200
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

.. Index file for the Python bob::machine bindings

==========
 Machines
==========

Resources for projecting, estimating probabilities or classifying data.

.. module:: bob.machine
   
.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   linear_scoring
   roll
   tnorm
   unroll
   znorm
   ztnorm
   ztnorm_same_value

.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   Activation
   BICMachine
   GMMMachine
   GMMStats
   GaborGraphMachine
   GaborJetSimilarity
   Gaussian
   HyperbolicTangentActivation
   ISVBase
   ISVMachine
   IVectorMachine
   IdentityActivation
   JFABase
   JFAMachine
   KMeansMachine
   LinearActivation
   LinearMachine
   LogisticActivation
   MLP
   MachineDoubleBase
   MachineGMMStatsA1DBase
   MachineGMMStatsScalarBase
   MultipliedHyperbolicTangentActivation
   PLDABase
   PLDAMachine
   SVMFile
   SupportVector
   WienerMachine

.. rubric:: Enumerations

.. autosummary::
   :toctree: generated/

   gabor_jet_similarity_type
   svm_kernel_type
   svm_type
