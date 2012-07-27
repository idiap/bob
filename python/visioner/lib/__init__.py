#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch> 
# Sun 24 Jul 17:50:01 2011 CEST

from ._visioner import *
from os import path

DEFAULT_DETECTION_MODEL = path.join(path.dirname(__file__), 'detection.gz')
"""Default classification model for basic face detection"""

DEFAULT_LOCALIZATION_MODEL = path.join(path.dirname(__file__), 
  'localization.gz')
"""Default keypoint localization model. TODO: How many points?"""

class MaxDetector(CVDetector):
  """A class that bridges the Visioner to bob so as to detect the most
  face-like object in still images or video frames"""

  def __init__(self, model_file=None, threshold=0.0, scanning_levels=0, 
      scale_variation=2, clustering=0.05,
      method=DetectionMethod.GroundTruth):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    model
      file containing the model to be loaded; **note**: Serialization will use a native text format by default. Files that have their names suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.
      
    threshold
      object classification threshold
      
    scanning_levels
      scanning levels (the more, the faster)
      
    scale_variation
      scale variation in pixels
      
    clustering
      overlapping threshold for clustering detections
      
    method
      Scanning or GroundTruth
    """

    if model_file is None: model_file = DEFAULT_DETECTION_MODEL

    CVDetector.__init__(self, model_file, threshold, scanning_levels,
        scale_variation, clustering, method)

  def __call__(self, grayimage):
    """Runs the detection machinery, returns a single bounding box

    Keyword parameters:

    image
      A gray-scaled image (2D array) with dtype=uint8.

    Returns a single (highest scored) detection as a bounding box.
    """

    return self.detect_max(grayimage)

class Detector(CVDetector):
  """A class that bridges the Visioner to bob so as to detect faces in 
  still images or video frames"""

  def __init__(self, model_file=None, threshold=0.0, scanning_levels=0, 
      scale_variation=2, clustering=0.05,
      method=DetectionMethod.GroundTruth):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    model
      file containing the model to be loaded; **note**: Serialization will use a native text format by default. Files that have their names suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.
      
    threshold
      object classification threshold
      
    scanning_levels
      scanning levels (the more, the faster)
      
    scale_variation
      scale variation in pixels
      
    clustering
      overlapping threshold for clustering detections
      
    method
      Scanning or GroundTruth
    """

    if model_file is None: model_file = DEFAULT_DETECTION_MODEL

    CVDetector.__init__(self, model_file, threshold, scanning_levels,
        scale_variation, clustering, method)

  def __call__(self, grayimage):
    """Runs the detection machinery, returns all bounding boxes above
    threshold. Detections are already clustered following the clustering
    parameter. The iterable contains detections in descending order with the
    first being the one with the highest score.

    Keyword parameters:

    image
      A gray-scaled image (2D array) with dtype=uint8.

    Returns an iterable with all detected bounding boxes in descending score
    order (first one is has the highest score).
    """

    return self.detect(grayimage)

class Localizer(CVLocalizer):
  """A class that bridges the Visioner to bob to localize keypoints in 
  still images or video frames"""

  def __init__(self, model_file=None,
      method=LocalizationMethod.MultipleShots_Median,
      detector=None):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    model_file
      Path to a file containing the keypoint localization model. If None is
      given, use the default localizer.

    method
      SingleShot, MultipleShots_Average or MultipleShots_Median

    detector
      Path to a file or a CVDetector (or Max/Detector) object to be used as the
      basis for the localization procedure. If None is given, use the default
      detector. 
    """

    if model_file is None: model_file = DEFAULT_LOCALIZATION_MODEL
    CVLocalizer.__init__(self, model_file, method)

    if detector is None: 
      self.detector = Detector()
    elif isinstance(detector, (str, unicode)): 
      self.detector = Detector(detector)
    elif isinstance(detector, CVDetector):
      self.detector = detector
    else:
      raise RuntimeError, 'input detector has to be either None, a file path or Detector object'

  def __call__(self, image):
    """Runs the localization machinery, returns the bounding box and points
 
    Keyword parameters:

    image
      A gray-scaled image (2D array) with dtype=uint8.

    Returns a bounding box and a set of keypoints.
    """

    return self.locate(self.detector, grayimage)

__all__ = dir()
