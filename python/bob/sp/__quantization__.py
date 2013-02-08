from ._sp import __quantization_uint8__
from ._sp import __quantization_uint16__

import numpy

class Quantization:
  """
  Objects of this class, after configuration, can quantize 1D or 2D signal into different number of levels. At the moment, only uint8 and uint16 input signals are supported.
  """
  
  """Class attributes"""
  
  @property
  def quantization_type(self):
    'Possible types of quantization: "uniform" (uniform quantization of the input signal within the range between min_level and max_level); "uniform_rounding" (uniform quantization of the input signal within the range between min_level and max_level, but similar to Matlab quantization (see http://www.mathworks.com/matlabcentral/newsreader/view_thread/275291); "user_spec" (quantization according to user-specified quantization table of thresholds.)'
    return self._quantization_type
  
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
  
  
  def __init__(self, dtype, quantization_type=None, num_levels=None, min_level=None, max_level=None, quantization_table=None):
    """
    Constructor. 
    
      dtype
        Data type (eg. numpy.dtype object. Supported data types are uint8 and uint16
        
      quantization_type
        Possible values of this parameter: "uniform" (uniform quantization of the input signal within the range between min_level and max_level); "uniform_rounding" (uniform quantization of the input signal within the range between min_level and max_level, but similar to Matlab quantization (see http://www.mathworks.com/matlabcentral/newsreader/view_thread/275291); "user_spec" (quantization according to user-specified quantization table of thresholds.)
        
      num_levels
        Number of quantization levels. The default is the total number of discreet values permitted by the dtype
        
      min_level
        Input values smaller than or equal to this value are scaled to this value prior to quantization. As a result, they will be scaled in the lowest qunatization level. The default is the minimum value permitted by dtpe "  
        
      max_level
        Input values greater then this value are scaled to this value prior to quantization. As a result, they will be quantized in the highest quantization level. The default is the maximum value permitted by dtype.      
        
      quantization_table
        1D numpy.ndarray containing user-specified thresholds of the quantization. If this parameter is given, quantization_type is automatically set to "user_spec". Otherwise, it is recalculated according to other quantization parameters. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.
    """
    
    quant_type_dict = {'uniform':1, 'uniform_rounding':2, 'user_spec':3}
    quant_type_invdict = {1:'uniform', 2:'uniform_rounding', 3:'user_spec'}
    
    if quantization_type != None and quantization_type not in quant_type_dict.keys():
      raise RuntimeError, 'Unknown quantization type'  
    
    dt = numpy.dtype(dtype)
    if dt == numpy.uint8:
      
      if (quantization_type != None and num_levels != None and min_level == None and max_level == None and quantization_table == None): 
        self.Q = __quantization_uint8__(quant_type_dict[quantization_type], num_levels)
      elif (quantization_type != None and num_levels != None and min_level != None and max_level != None and quantization_table == None):
        self.Q = __quantization_uint8__(quant_type_dict[quantization_type], num_levels, min_level, max_level)
      elif (quantization_type == None and num_levels == None and min_level == None and max_level == None and quantization_table != None):
        self.Q = __quantization_uint8__(quantization_table)  
      elif (quantization_type == None and num_levels == None and min_level == None and max_level == None):
        self.Q = __quantization_uint8__()
      else:
        raise RuntimeError, 'Unknown configuration for creating a Quantization object'  
    
    elif dt == numpy.uint16:  
      if (quantization_type != None and num_levels != None and min_level == None and max_level == None and quantization_table == None): 
        self.Q = __quantization_uint16__(quant_type_dict[quantization_type], num_levels)
      elif (quantization_type != None and num_levels != None and min_level != None and max_level != None and quantization_table == None):
        self.Q = __quantization_uint16__(quant_type_dict[quantization_type], num_levels, min_level, max_level)
      elif (quantization_type == None and num_levels == None and min_level == None and max_level == None and quantization_table != None):
        self.Q = __quantization_uint16__(quantization_table)  
      elif (quantization_type == None and num_levels == None and min_level == None and max_level == None and quantization_table == None):
        self.Q = __quantization_uint16__()
      else:
        raise RuntimeError, 'Not known configuration for creating a Quantization object'  

    else:
      raise RuntimeError, 'Quantization does not support data of type ', dt   
      
    self._quantization_table = self.Q.thresholds     
    self._quantization_type = quant_type_invdict[self.Q.type]     
    self._min_level = self.Q.min_level
    self._max_level = self.Q.max_level 
    self._num_levels = self.Q.num_levels    

  

  def __call__(self, input_signal):
    """
    Performs quantization of a 1D or 2D signal.
    """
    output_signal = numpy.ndarray(input_signal.shape, 'uint32');
    self.Q.__call__(input_signal, output_signal)
    return output_signal
    
  def quantization_level(self, input_value):
    """
    Returns the quantization level of a single input-value
    """
    return self.Q.quantization_level(input_value)  

__all__ = dir()      
           
           
