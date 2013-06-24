from ..core import __from_extension_import__
__from_extension_import__('._machine', __package__, locals())

from . import __linear__
from . import __mlp__

def ztnorm_same_value(vect_a, vect_b):
  """Computes the matrix of boolean D for the ZT-norm, which indicates where 
     the client ids of the T-Norm models and Z-Norm samples match.

     vect_a An (ordered) list of client_id corresponding to the T-Norm models
     vect_b An (ordered) list of client_id corresponding to the Z-Norm impostor samples
  """
  from .. import core
  import numpy as np
  sameMatrix = np.ndarray((len(vect_a), len(vect_b)), 'bool')
  for j in range(len(vect_a)):
    for i in range(len(vect_b)):
      sameMatrix[j, i] = (vect_a[j] == vect_b[i])
  return sameMatrix

__all__ = [k for k in dir() if not k.startswith('_')]
