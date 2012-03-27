#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# 12-02-27

"""Extensions of the bob.ip.GaborWaveletTransform class
"""

from ._ip import GaborWaveletTransform
from ._ip import rgb_to_gray
import numpy

def gwt_trafo_image(self,input_image):
  """ This function creates an empty trafo image for the given input image.
  Use this function to generate the trafo image in the correct size and with the correct data type.
  In case you have to transform multiple images of the same size, this trafo image can be reused.
  """
  return numpy.ndarray([self.number_of_kernels, input_image.shape[-2], input_image.shape[-1]], dtype=complex)
#register function
GaborWaveletTransform.empty_trafo_image = gwt_trafo_image
del gwt_trafo_image


def gwt_transform(self,input_image,output_trafo_image):
  """ This function performs a Gabor wavelet transform of any kind of images 
  gray/color and int8/double/... and saves the result in the given trafo image
  """
  # cast image into complex type
  if input_image.ndim == 2:
    # gray image; convert to complex
    input_ = input_image.astype(complex)
  elif input_image.ndim == 3:
    # color image; color convert first
    gray_image = numpy.ndarray(input_image.shape[1:], input_image.dtype)
    rgb_to_gray(input_image, gray_image)
    input_ = gray_image.astype(complex)
  else:
    bob.core.throw_exception()
  
  # call the GWT
  self.perform_gwt(input_,output_trafo_image)
#register function
GaborWaveletTransform.__call__ = gwt_transform
del gwt_transform



def gwt_jet_image(self, input_image, include_phases = True):
  """ This function creates an empty jet image (with or without phases) for the given input image.
  Use this function to generate the jet image in the correct size and with the correct data type.
  In case you have to transform multiple images of the same size, this jet image can be reused.
  """
  if include_phases:
    return numpy.ndarray([input_image.shape[-2], input_image.shape[-1], 2, self.number_of_kernels], dtype=numpy.float64)
  else:
    return numpy.ndarray([input_image.shape[-2], input_image.shape[-1], self.number_of_kernels], dtype=numpy.float64)
    
#register function
GaborWaveletTransform.empty_jet_image = gwt_jet_image
del gwt_jet_image


def gwt_compute_jets(self, input_image, output_jet_image, normalize=True):
  """ This function performs a Gabor wavelet transform of any kind of images 
  gray/color and int8/double/... and fills the Gabor jets of the given jet image.
  """
  # cast image into complex type
  if input_image.ndim == 2:
    # gray image; convert to complex
    input_ = input_image.astype(complex)
  elif input_image.ndim == 3:
    # color image; color convert first
    gray_image = numpy.ndarray(input_image.shape[1:], input_image.dtype)
    rgb_to_gray(input_image, gray_image)
    input_ = gray_image.astype(complex)
  else:
    bob.core.throw_exception()
  
  # call the GWT
  self.compute_jet_image(input_,output_jet_image, normalize)
#register function
GaborWaveletTransform.compute_jets = gwt_compute_jets
del gwt_compute_jets
