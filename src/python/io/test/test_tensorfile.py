#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Tests Torch support to read and write .tensor file.
"""

import os, sys
import unittest
import tempfile
import torch
import random

def get_tempfilename(prefix='torchtest_', suffix='.tensor'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

class TensorFileTest(unittest.TestCase):
  """Performs various tests for the Torch::io::TensorFile types."""
 
  def test01_CanCreate(self):
    # This test demonstrates how to create Torch binary files from scratch,
    # starting from blitz::Arrays

    # We start by creating some arrays to play with. Please note that in normal
    # cases you are either generating these arrays or reading from other binary
    # files or datasets.
    N = 200
    SHAPE = (2, 4, 8) #64 elements in each array
    NELEMENT = SHAPE[0] * SHAPE[1] * SHAPE[2]
    arrays = []
    for k in range(N):
      data = [random.uniform(0,10) for z in range(NELEMENT)] 
      arrays.append(numpy.array(data, 'float64').reshape(SHAPE))

    # Now we create a new binary output file in a temporary location and save
    # the data there.
    tmpname = get_tempfilename()
    outfile = torch.io.TensorFile(tmpname, 'w')
    for k in arrays: outfile.write(k)

    # We can verify some of the output file's properties
    self.assertEqual(len(outfile), N)
    self.assertEqual(outfile.shape, SHAPE)
    self.assertEqual(outfile.elementType, torch.io.ElementType.float64)
    del outfile

    # Now we open the binary file for reading and the data should be all there.
    # Loading the data should prove us it is as we expect it
    infile = torch.io.TensorFile(tmpname, 'r')
    self.assertEqual(len(infile), N)
    self.assertEqual(infile.shape, SHAPE)
    self.assertEqual(infile.elementType, torch.io.ElementType.float64)
    for k in range(N): 
      self.assertEqual(infile[k], arrays[k])
    del infile

    # Now we open the tensor file for reading and the data should be all there.
    # We try both flags at the same time
    inoutfile = torch.io.TensorFile(tmpname, 'rw') 
    self.assertEqual(len(inoutfile), N)
    self.assertEqual(inoutfile.shape, SHAPE)
    self.assertEqual(inoutfile.elementType, torch.io.ElementType.float64)
    for k in range(N): 
      self.assertEqual(inoutfile[k], arrays[k])
    del inoutfile
    os.unlink(tmpname)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.chdir('data')
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

