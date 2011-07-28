#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch> 
# Sun 24 Jul 17:50:01 2011 CEST

from libpytorch_visioner import *
from os import path

DEFAULT_CMODEL = path.join(path.dirname(__file__), 'Face.MCT9')
DEFAULT_LMODEL = path.join(path.dirname(__file__), 'Facial.MCT9.TMaxBoost')

class MaxDetector:
  """A class that bridges the Visioner to torch so as to detect the most
  face-like object in still images or video frames"""

  def __init__(self, cmodel_file=None, scan_levels=0, scale_var=8):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    cmodel_file
      Path to a file containing the object classification model. If unset (or
      set to None), I will use the default model file installed with the
      release.

    scan_levels
      scanning levels (the more, the faster)

    scale_var
      scanning: scale variation in pixels
    """

    if cmodel_file is None: cmodel_file = DEFAULT_CMODEL

    self.cmodel, self.cparam = load_model(cmodel_file)
    self.cparam.ds = scale_var
    self.cscanner = SWScanner(self.cparam)
    self.scan_levels = scan_levels

  def __call__(self, grayimage):
    """Runs the detection machinery, returns bounding boxes"""

    self.cscanner.load(grayimage)
    return detect_max(self.cmodel, self.scan_levels, self.cscanner)

class Detector:
  """A class that bridges the Visioner to torch so as to detect faces in 
  still images or video frames"""

  def __init__(self, cmodel_file=None, threshold=0., 
      scan_levels=0, scale_var=8, cluster=0.10):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    cmodel_file
      Path to a file containing the object classification model. If unset (or
      set to None), I will use the default model file installed with the
      release.

    threshold
      Classifier threshold

    scan_levels
      scanning levels (the more, the faster)

    scale_var
      scanning: scale variation in pixels
    
    cluster
      NMS clustering: overlapping threshold
    """

    if cmodel_file is None: cmodel_file = DEFAULT_CMODEL

    self.cmodel, self.cparam = load_model(cmodel_file)
    self.cparam.ds = scale_var
    self.cscanner = SWScanner(self.cparam)
    self.threshold = threshold
    self.scan_levels = scan_levels
    self.cluster = cluster

  def __call__(self, grayimage):
    """Runs the detection machinery, returns bounding boxes"""

    self.cscanner.load(grayimage)
    return detect(self.cmodel, self.threshold, self.scan_levels, 
        self.cluster, self.cscanner)

class Localizer:
  """A class that bridges the Visioner to torch so as to localize face in 
  still images or video frames"""

  def __init__(self, cmodel_file=None, lmodel_file=None, scan_levels=0, scale_var=8):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    cmodel_file
      Path to a file containing the object classification model. If unset (or
      set to None), I will use the default model file installed with the
      release.

    lmodel_file
      Path to a file containing the keypoints localization model. If unset (or
      set to None), I will use the default model file installed with the
      release.

    scan_levels
      scanning levels (the more, the faster)

    scale_var
      scanning: scale variation in pixels
    """

    if cmodel_file is None: cmodel_file = DEFAULT_CMODEL
    if lmodel_file is None: lmodel_file = DEFAULT_LMODEL

    self.cmodel, self.cparam = load_model(cmodel_file)
    self.cparam.ds = scale_var
    self.cscanner = SWScanner(self.cparam)
    self.lmodel, self.lparam = load_model(lmodel_file)
    self.lparam.ds = scale_var
    self.lscanner = SWScanner(self.lparam)
    self.scan_levels = scan_levels

  def __call__(self, grayimage):
    """Runs the localization machinery, returns points"""

    self.cscanner.load(grayimage)
    self.lscanner.load(grayimage)
    return locate(self.cmodel, self.lmodel, self.scan_levels, 
        self.cscanner, self.lscanner)

__all__ = dir()
