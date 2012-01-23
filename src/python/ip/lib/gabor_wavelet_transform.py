#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# \today 

"""Extensions of the GWT class
"""

from libpybob_ip import GaborWaveletTransform
from libpybob_ip import rgb_to_gray
import numpy

def gwt_trafo_image(self,input_image):
  """ This function creates an empty trafo image for the given input image
  Use this function to generate the trafo image in the correct size and with the correct data type.
  In case you have to transform multiple images of the same size, this trafo image can be reused.
  """
  return numpy.ndarray([self.number_of_kernels, input_image.shape[-2], input_image.shape[-1]], dtype=complex)
#register function
GaborWaveletTransform.trafo_image = gwt_trafo_image
del gwt_trafo_image


def gwt_transform(self,input_image,output_trafo_image):
  """ This function performs a Gabor wavelet transform to any kind of images 
  gray/color and int8/double/...  
  """
  # cast image into complex type
  if input_image.ndim == 2:
    # gray image; convert to complex
    input_ = input_image.astype(complex)
  elif input_image.ndim == 3:
    # color image; color convert first
    input_ = rgb_to_gray(input_image).astype(complex)
  else:
    bob.core.throw_exception()
  
  # call the GWT
  self.perform_gwt(input_,output_trafo_image)
#register function
GaborWaveletTransform.transform = gwt_transform
del gwt_transform



def gwt_jet_image(self,input_image):
  """ This function creates an empty jet image for the given input image.
  Use this function to generate the jet image in the correct size and with the correct data type.
  In case you have to transform multiple images of the same size, this jet image can be reused.
  """
  return numpy.ndarray([input_image.shape[-2], input_image.shape[-1], 2, self.number_of_kernels], dtype=numpy.float64)
#register function
GaborWaveletTransform.jet_image = gwt_jet_image
del gwt_jet_image


def gwt_compute_jets(self,input_image, output_jet_image, normalize=True):
  """ This function performs a Gabor wavelet transform to any kind of images 
  gray/color and int8/double/... and fills the Gabor jets
  """
  # cast image into complex type
  if input_image.ndim == 2:
    # gray image; convert to complex
    input_ = input_image.astype(complex)
  elif input_image.ndim == 3:
    # color image; color convert first
    input_ = rgb_to_gray(input_image).astype(complex)
  else:
    bob.core.throw_exception()
  
  # call the GWT
  self.compute_jet_image(input_,output_jet_image, normalize)
#register function
GaborWaveletTransform.compute_jets = gwt_compute_jets
del gwt_compute_jets
