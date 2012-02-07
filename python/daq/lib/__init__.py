from ._cxx import *
from os import path
import types

# Get the default model from visioner package
DEFAULT_CMODEL = path.join(path.dirname(__file__), '../visioner/Face.MCT9.gz')

###
# We want to change the constructors to make the model an optional argument
###

# Save the original constructor
CaptureSystem._old_init = CaptureSystem.__init__
VisionerFaceLocalization._old_init = VisionerFaceLocalization.__init__

# Declare our new constructors
def _CaptureSystem__init__(self, camera, model = DEFAULT_CMODEL):
  self._old_init(camera, model)

def _VisionerFaceLocalization__init__(self, model = DEFAULT_CMODEL):
  self._old_init(model)

# Replace the original constructors with the new one
CaptureSystem.__init__ = _CaptureSystem__init__
VisionerFaceLocalization.__init__ = _VisionerFaceLocalization__init__

__all__ = [d for d in dir() if d[:1] != '_']
