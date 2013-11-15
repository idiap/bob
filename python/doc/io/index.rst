.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 20 Jun 08:04:20 2013 CEST
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

.. Index file for the Python bob::io bindings

===================
 Data Input/Output
===================

Input and output facilities.

.. module:: bob.io

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   append
   create_directories_save
   extensions
   load
   merge
   peek
   peek_all
   save
   write

.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   File
   HDF5Descriptor
   HDF5File
   HDF5Type
   open

.. rubric:: Video Handling

.. autosummary::
   :toctree: generated/

   available_video_codecs
   available_videoreader_formats
   available_videowriter_formats
   describe_video_decoder
   describe_video_encoder
   supported_video_codecs
   supported_videoreader_formats
   supported_videowriter_formats
   VideoReaderIterator
   VideoReader
   VideoWriter
