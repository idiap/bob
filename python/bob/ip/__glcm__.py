from ._ip import __GLCM_uint8__
from ._ip import __GLCM_uint16__

import numpy

class GLCM(object):
  """
  Objects of this class, after configuration, can calculate GLCM matrix on a given input image. At the moment, only uint8 and uint16 input images are supported.
  """
  
  """Class properties"""
  
  @property
  def num_levels(self):
    'Number of quantization levels.'
    return self._num_levels
    
  @property
  def max_level(self):
    'Input values greater then this value are scaled to this value prior to quantization. As a result, they will be quantized in the highest quantization level.'
    return self._max_level

  @property
  def min_level(self):
    'Input values smaller than or equal to this value are scaled to this value prior to quantization. As a result, they will be scaled in the lowest qunatization level.'
    return self._min_level
  
  @property
  def quantization_table(self):
    '1D numpy.ndarray containing user-specified thresholds of the quantization. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.'
    return self._quantization_table
    
  @property
  def symmetric(self):
    'If True, the output matrix for each specified distance and angle will be symmetric. Both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.'  
    return self._symmetric
    
  @symmetric.setter
  def symmetric(self, value):
    self._symmetric = value
    self.G.symmetric = value
  
  @property
  def normalized(self):
    'If True, each matrix for each specified distance and angle will be normalized by dividing by the total number of accumulated co-occurrences. The default is False.'
    return self._normalized
    
  @normalized.setter
  def normalized(self, value):
    self._normalized = value
    self.G.normalized = value
    
  @property
  def offset(self):
    "2D numpy.ndarray of dtype='int32' specifying the column and row distance between pixel pairs. The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM."
    return self._offset    
  
  @offset.setter
  def offset(self, value):
    self._offset = value
    self.G.offset = value
  
    
  def __init__(self, dtype, num_levels=None, min_level=None, max_level=None, quantization_table=None):
    """
    Constructor. 
    
      dtype
        Data type (eg. numpy.dtype object. Supported data types are uint8 and uint16
        
      num_levels
        Number of quantization levels. The default is the total number of discreet values permitted by dtype
        
      min_level
        Input values smaller than or equal to this value are scaled to this value prior to quantization. As a result, they will be scaled in the lowest qunatization level. The default is the minimum value permitted by dtype "  
        
      max_level
        Input values greater then this value are scaled to this value prior to quantization. As a result, they will be quantized in the highest quantization level. The default is the maximum value permitted by dtype.      
        
      quantization_table
        1D numpy.ndarray containing user-specified thresholds of the quantization. If not given, it is recalculated according to other quantization parameters. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.
    """
    
    dt = numpy.dtype(dtype)
    if dt == numpy.uint8:
      
      if (num_levels != None and min_level == None and max_level == None and quantization_table == None): 
        self.G = __GLCM_uint8__(num_levels)
      elif (num_levels != None and min_level != None and max_level != None and quantization_table == None):
        self.G = __GLCM_uint8__(num_levels, min_level, max_level)
      elif (num_levels == None and min_level == None and max_level == None and quantization_table != None):
        self.G = __GLCM_uint8__(quantization_table)  
      elif (num_levels == None and min_level == None and max_level == None):
        self.G = __GLCM_uint8__()
      else:
        raise RuntimeError, 'Unknown configuration for creating GLCM object'  
    
    elif dt == numpy.uint16:  
      if (num_levels != None and min_level == None and max_level == None and quantization_table == None): 
        self.G = __GLCM_uint16__(num_levels)
      elif (num_levels != None and min_level != None and max_level != None and quantization_table == None):
        self.G = __GLCM_uint16__(num_levels, min_level, max_level)
      elif (num_levels == None and min_level == None and max_level == None and quantization_table != None):
        self.G = __GLCM_uint16__(quantization_table)  
      elif (num_levels == None and min_level == None and max_level == None):
        self.G = __GLCM_uint16__()
      else:
        raise RuntimeError, 'Unknown configuration for creating GLCM object'  

    else:
      raise RuntimeError, 'GLCM does not support data of type ', dt   
      
    self._offset = self.G.offset
    self._quantization_table = self.G.quantization_table
    self._min_level = self.G.min_level
    self._max_level = self.G.max_level 
    self._num_levels = self.G.num_levels    
    self._symmetric = self.G.symmetric
    self._normalized = self.G.normalized


  def __call__(self, input_image):
    """
    Calculates GLCM on a given input image
    """
    glcm = numpy.ndarray(self.G.get_glcm_shape(), 'double');
    self.G.__call__(input_image, glcm)
    return glcm
    
  def get_glcm_shape(self):
    """ 
    Returns the shape of the output GLCM.
    """
    return self.G.get_glcm_shape()

__all__ = dir()      
           
           
