#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os, sys
import unittest
import bob
import numpy
import array
import math
import time

#############################################################################
# Tests blitz-based extrapolation implementation with values returned 
#############################################################################

########################## Values used for the computation ##################
eps = 1e-3

#############################################################################
numpy.set_printoptions(precision=2, threshold=numpy.nan, linewidth=200)
  
def _read(filename):
  """Read video.FrameContainer containing preprocessed frames"""

  fileName, fileExtension = os.path.splitext(filename)
  wav_filename = filename
  import scipy.io.wavfile
  rate, data = scipy.io.wavfile.read(str(wav_filename)) # the data is read in its native format
  if data.dtype =='int16':
    data = numpy.cast['float'](data)
  return [rate,data]


def compare(v1, v2, width):
  return abs(v1-v2) <= width
  
def mel_python(f):
  import math
  return 2595.0*math.log10(1.+f/700.0)

def mel_inv_python(value):
  return 700.0 * (10 ** (value / 2595.0) - 1)

def sig_norm(win_length, frame, flag):
  gain = 0.0
  for i in range(win_length):
    gain = gain + frame[i] * frame[i]
  
  ENERGY_FLOOR = 1.0
  if gain < ENERGY_FLOOR:
    gain = math.log(ENERGY_FLOOR)
  else:
    gain = math.log(gain)
    
  if(flag and gain != 0.0):
    for i in range(win_length):
      frame[i] = frame[i] / gain
  return gain

def pre_emphasis(frame, win_length, a):
  if (a < 0.0) or (a >= 1.0):
    print "Error: The emphasis coeff. should be between 0 and 1"
  if (a == 0.0):
    return frame
  else:
    for i in range(win_length - 1, 0, -1):
      frame[i] = frame[i] - a * frame[i - 1]
    frame[0] = (1. - a) * frame[0]  
  return frame

def hamming_window(vector, hamming_kernel, win_length):
  for i in range(win_length):
    vector[i] = vector[i] * hamming_kernel[i]
  return vector

def log_filter_bank(x, n_filters, p_index, win_size):
  x1 = numpy.array(x, dtype=numpy.complex128)
  complex_ = bob.sp.fft(x1)
  for i in range(0, win_size / 2 + 1):
    re = complex_[i].real
    im = complex_[i].imag
    x[i] = math.sqrt(re * re + im * im) 
  filters = log_triangular_bank(x, n_filters, p_index)
  return filters

def log_triangular_bank(data, n_filters, p_index):
  a = 1.0 / (p_index[1:n_filters+2] - p_index[0:n_filters+1] + 1)
  vec1 =  list(numpy.arange(p_index[i], p_index[i + 1]) for i in range(0, n_filters))
  vec2 =  list(numpy.arange(p_index[i+1], p_index[i + 2] + 1) for i in range(0, n_filters))
  res_ = numpy.array([(numpy.sum(data[vec1[i]]*(1.0 - a [i]* (p_index[i + 1]-(vec1[i])))) + 
          numpy.sum(data[vec2[i]] * (1.0 - a[i+1] * ( (vec2[i]) - p_index[i + 1])))) 
          for i in range(0, n_filters)])
  FBANK_OUT_FLOOR = 1.0
  filters = numpy.log(numpy.where(res_ < FBANK_OUT_FLOOR, FBANK_OUT_FLOOR, res_))
  return filters

def dct_transform(filters, n_filters, dct_kernel, n_ceps, dct_norm):
  if dct_norm:
    dct_coeff = numpy.sqrt(2.0/(n_filters))
  else :
    dct_coeff = 1.0

  ceps = numpy.zeros(n_ceps + 1)
  vec = numpy.array(range(1, n_filters + 1))
  for i in range(1, n_ceps + 1):
    ceps[i - 1] = numpy.sum(filters[vec - 1] * dct_kernel[i - 1][0:n_filters])
    ceps[i - 1] = ceps[i - 1] * dct_coeff
    
  return ceps

def cepstral_features_extraction(obj, rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta):
  #########################
  ## Initialisation part ##
  #########################
  c = bob.ap.Ceps(rate_wavsample[0], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef)
  c.dct_norm = dct_norm
  c.mel_scale = mel_scale
  c.with_energy = with_energy
  c.with_delta = with_delta
  c.with_delta_delta = with_delta_delta
  #ct = bob.ap.TestCeps(c)

  sf = rate_wavsample[0]
  data = rate_wavsample[1]

  win_length = int (sf * win_length_ms / 1000)
  win_shift = int (sf * win_shift_ms / 1000)
  win_size = int (2.0 ** math.ceil(math.log(win_length) / math.log(2)))
  m = int (math.log(win_size) / math.log(2))

  # Hamming initialisation 
  cst = 2 * math.pi / (win_length - 1.0)
  hamming_kernel = numpy.zeros(win_length)
  
  for i in range(win_length):
    hamming_kernel[i] = (0.54 - 0.46 * math.cos(i * cst))

  # Compute cut-off frequencies 
  p_index = numpy.array(numpy.zeros(n_filters + 2), dtype=numpy.int16)
  if(mel_scale):
    # Mel scale
    m_max = mel_python(f_max)
    #obj.assertAlmostEqual(ct.herz_to_mel(f_max), m_max, 7, "Error in Mel...")
    m_min = mel_python(f_min)
    #obj.assertAlmostEqual(ct.herz_to_mel(f_min), m_min, 7, "Error in Mel...")
  
    for i in range(n_filters + 2):
      alpha = ((i) / (n_filters + 1.0))
      f = mel_inv_python(m_min * (1 - alpha) + m_max * alpha)
      #obj.assertAlmostEqual(ct.mel_to_herz(m_min * (1 - alpha) + m_max * alpha), f, 7, "Error in MelInv...")
      factor = f / (sf * 1.0)
      p_index[i] = int (round((win_size) * factor))
  else:
    #linear scale
    for i in range(n_filters + 2):
      alpha = (i) / (n_filters + 1.0)
      f = f_min * (1.0 - alpha) + f_max * alpha
      p_index[i] = int (round((win_size / (sf * 1.0) * f)))

  #Cosine transform initialisation
  dct_kernel = [ [ 0 for i in range(n_filters) ] for j in range(n_ceps) ] 
  
  for i in range(1, n_ceps + 1):
    for j in range(1, n_filters + 1):
      dct_kernel[i - 1][j - 1] = math.cos(math.pi * i * (j - 0.5) / n_filters)

  ######################################
  ### End of the Initialisation part ###
  ######################################
  
  ######################################
  ###          Core code             ###
  ######################################

  data_size = data.shape[0]
  n_frames = 1 + (data_size - win_length) / win_shift
 
  # create features set
  ceps_sequence = numpy.zeros(n_ceps)
  dim0 = n_ceps
  if(with_energy):
    dim0 += + 1
  dim = dim0
  if(with_delta):
    dim += dim0
    if(with_delta_delta):
      dim += dim0
  else:
    with_delta_delta = False
  
  params = [ [ 0 for i in range(dim) ] for j in range(n_frames) ] 
   
  # compute cepstral coefficients
  delta = 0
  for i in range(n_frames):
    # create a frame
    frame = numpy.zeros(win_size, dtype=numpy.float64)
    som = 0.0
    vec = numpy.arange(win_length)  
    frame[vec] = data[vec + i * win_shift]
    som = numpy.sum(frame)
    som = som / win_size
    frame = frame - som  
    
    if (with_energy):
      energy = sig_norm(win_length, frame, False)
      #e1 = ct.log_energy(frame)
      #obj.assertAlmostEqual(e1, energy, 7, "Error in Energy Computation...")
    
    f2 = numpy.copy(frame)  
    
    # pre-emphasis filtering
    frame = pre_emphasis(frame, win_length, pre_emphasis_coef)
    #ct.pre_emphasis(f2)
    #for kk in range(len(frame)):
    #  obj.assertAlmostEqual(frame[kk], f2[kk], 7, "Error in Pre-Emphasis Computation...")
    
    # Hamming windowing
    f2 = numpy.copy(frame)
    frame = hamming_window(frame, hamming_kernel, win_length)
    #ct.hamming_window(f2)
    #for kk in range(len(frame)):
    #  obj.assertAlmostEqual(frame[kk], f2[kk], 7, "Error in Pre-Emphasis Computation...")
    
    f2=numpy.copy(frame)
    filters = log_filter_bank(frame, n_filters, p_index, win_size)

    #filt2 = ct.log_filter_bank(f2, win_size, n_filters)
    
    #for kk in range(len(filters)):
    #  obj.assertAlmostEqual(filters[kk], filt2[kk], 7, "Error in log Filtering")
          
    ceps = dct_transform(filters, n_filters, dct_kernel, n_ceps, dct_norm)
    #ceps2 = ct.apply_dct(n_ceps)


    if(with_energy):
      d1 = n_ceps + 1
      ceps[n_ceps] = energy
        #print ceps
    else:
      d1 = n_ceps

    # stock the results in params matrix
    vec=numpy.arange(d1)
    params[i][0:d1]=ceps[vec]

  # compute Delta coefficient
  if(with_delta):
    som = 0.0
    for i in range(1,delta_win+1):
      som = som + i*i
    som = som *2
         
    for i in range(n_frames):
      for k in range(n_ceps):
        params[i][d1+k] = 0.0
        for l in range(1, delta_win+1):
          if (i+l < n_frames):
            p_ind = i+l
          else:
            p_ind = n_frames - 1
          if (i-l > 0):
            n_ind = i-l
          else:
            n_ind = 0
          params[i][d1+k] = params[i][d1+k] + l * (params[p_ind][k] - params[n_ind][k])
        params[i][d1+k] = params[i][d1+k] / som

  # compute Delta of the Energy
  if(with_delta and with_energy):
    som = 0.0
    
    vec=numpy.arange(1,delta_win+1)
    som = 2.0* numpy.sum(vec*vec)
    
    for i in range(n_frames):
      k = n_ceps
      params[i][d1+k] = 0.0
      for l in range(1, delta_win+1):
        if (i+l < n_frames):
          p_ind = i+l
        else:
          p_ind = n_frames - 1
        if (i-l > 0):
          n_ind = i-l
        else:
          n_ind = 0
        params[i][d1+k] = params[i][d1+k] + l* (params[p_ind][k] - params[n_ind][k])
      params[i][d1+k] = params[i][d1+k] / som
   
  # compute Delta Delta of the coefficients
  if(with_delta_delta):
    som = 0.0
    for i in range(1,delta_win+1):
      som = som + i*i
    som = som *2
    for i in range(n_frames):
      for k in range(n_ceps):
        params[i][2*d1+k] = 0.0
        for l in range(1, delta_win+1):
          if (i+l < n_frames):
            p_ind = i+l
          else:
            p_ind = n_frames - 1
          if (i-l > 0):
            n_ind = i-l
          else:
            n_ind = 0
          params[i][2*d1+k] = params[i][2*d1+k] + l * (params[p_ind][d1+k] - params[n_ind][d1+k])
        params[i][2*d1+k] = params[i][2*d1+k] / som 
  
  # compute Delta Delta of the energy
  if(with_delta_delta and with_energy):
    som = 0.0
    for i in range(1,delta_win+1):
      som = som + i*i
    som = som *2
    for i in range(n_frames):
      k = n_ceps
      params[i][2*d1+k] = 0.0
      for l in range(1, delta_win+1):
        if (i+l < n_frames):
          p_ind = i+l
        else:
          p_ind = n_frames - 1
        if (i-l > 0):
          n_ind = i-l
        else:
          n_ind = 0
        params[i][2*d1+k] = params[i][2*d1+k] + l * (params[p_ind][d1+k] - params[n_ind][d1+k])
      params[i][2*d1+k] = params[i][2*d1+k] / som
  data = numpy.array(params)
  
  return data

def cepstral_comparison_run(obj, rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta):
  c = bob.ap.Ceps(rate_wavsample[0], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
  c.with_energy = with_energy
  c.with_delta = with_delta
  if c.with_delta:
    c.with_delta_delta = with_delta_delta
  #ct = bob.ap.TestCeps(c)
  A = c(rate_wavsample[1])
  B = cepstral_features_extraction(obj, rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, 
        f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
  diff=numpy.sum(numpy.sum((A-B)*(A-B)))
  obj.assertAlmostEqual(diff, 0., 7, "Error in Ceps Analysis")
    
##################### Unit Tests ##################  
class CepsTest(unittest.TestCase):
  """Test the Cepstral feature extraction"""

  def test_cepstral(self):
    import pkg_resources
    rate_wavsample = _read(pkg_resources.resource_filename(__name__, os.path.join('data', 'sample.wav')))
    
    win_length_ms = 20
    win_shift_ms = 10
    n_filters = 24
    n_ceps = 19
    f_min = 0.
    f_max = 4000.
    delta_win = 2
    pre_emphasis_coef = 0.97
    dct_norm = True
    mel_scale = True 
    with_energy = True
    with_delta = True
    with_delta_delta = True
    
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)

    # Varying win_length_ms
    win_length_ms = 30
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    # Varying win_shift_ms
    win_shift_ms = 15
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying n_filters
    n_filters = 20
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying n_ceps
    n_ceps = 12
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying f_min
    f_min = 300.
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying f_max
    f_max = 3300.
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying delta_win
    delta_win = 5
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying pre_emphasis_coef
    pre_emphasis_coef = 0.7
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying dct_norm
    dct_norm = False
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying mel_scale : linear scale
    mel_scale = False
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
    
    # Varying with_energy
    with_energy = False
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)

    # Varying with_delta
    with_delta = False
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
   
    # Varying with_delta_delta
    with_delta_delta = False
    cepstral_comparison_run(self,rate_wavsample, win_length_ms, win_shift_ms, n_filters, n_ceps, dct_norm, f_min, f_max, delta_win,
                               pre_emphasis_coef, mel_scale, with_energy, with_delta, with_delta_delta)
   
    # Test comparison operators and copy constructor
    c0 = bob.ap.Ceps(rate_wavsample[0], win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    c1 = bob.ap.Ceps(c0)
    c2 = bob.ap.Ceps(c1)
    c2.win_length_ms = 27. 
    self.assertTrue( c0 == c1)
    self.assertFalse(c0 != c1)
    self.assertFalse(c0 == c2)
    self.assertTrue( c0 != c2)
