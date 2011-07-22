from libpytorch_visioner import *

class Localizer:
  """A class that bridging the Visioner to torch so as to localize face in 
  still images or video frames"""

  def __init__(self, cmodel_file, lmodel_file, scan_levels=0, scale_var=8):
    """Creates a new face localization object by loading object classification
    and keypoint localization models from visioner model files.

    Keyword Parameters:

    cmodel_file
      File containing the object classification model

    lmodel_file
      File containing the keypoints localization model

    scan_levels
      scanning levels (the more, the faster)

    scale_var
      scanning: scale variation in pixels
    """

    self.cmodel, self.cparam = load_model(cmodel_file)
    self.cparam.ds = scale_var
    self.cscanner = SWScanner(self.cparam)
    self.lmodel, self.lparam = load_model(lmodel_file)
    self.lparam.ds = scale_var
    self.lscanner = SWScanner(self.lparam)
    self.scan_levels = scan_levels

  def __call__(self, grayimage):
    """Runs the localization machinery, returns points"""

    self.cmodel.load(grayimage)
    self.lmodel.load(grayimage)
    return locate(self.cmodel, self.lmodel, self.scan_levels, 
        self.cscanner, self.lscanner)

__all__ = dir()
