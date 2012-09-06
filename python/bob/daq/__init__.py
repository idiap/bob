from ._ext import *
from ..visioner import DEFAULT_DETECTION_MODEL
from os import path
import types

# Replace constructors with good defaults
CaptureSystem._old_init = CaptureSystem.__init__
def _CaptureSystem__init__(self, camera, model = DEFAULT_DETECTION_MODEL):
  self._old_init(camera, model)
CaptureSystem.__init__ = _CaptureSystem__init__
del _CaptureSystem__init__

VisionerFaceLocalization._old_init = VisionerFaceLocalization.__init__
def _VisionerFaceLocalization__init__(self, model = DEFAULT_DETECTION_MODEL):
  self._old_init(model)
VisionerFaceLocalization.__init__ = _VisionerFaceLocalization__init__
del _VisionerFaceLocalization__init__

__all__ = [d for d in dir() if d[:1] != '_']
