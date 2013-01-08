#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 13 Mar 2012 11:20:54 CET 

"""Video additions
"""

from . import _io

if hasattr(_io, "VideoReader"):
  
  # the VideoReader is an optional compile-in, so it may not exist on certain
  # installations.

  from ._io import VideoReader

  def load(self, raise_on_error=False):
    """Loads all of the video stream in a numpy ndarray organized in this way:
    (frames, color-bands, height, width). I'll dynamically allocate the output
    array and return it to you. 
    
    The flag ``raise_on_error``, which is set to ``False`` by default
    influences the error reporting in case problems are found with the video
    file. If you set it to ``True``, we will report problems raising
    exceptions. If you either don't set it or set it to ``False``, we will
    truncate the file at the frame with problems and will not report anything.
    It is your task to verify if the number of frames returned matches the
    expected number of frames as reported by the property ``number_of_frames``
    (or ``len``) of this object.
    """

    (frames, data) = self.__load__(raise_on_error=raise_on_error)
    data.resize(frames, data.shape[1], data.shape[2], data.shape[3])
    return data

  VideoReader.load = load
  del load
