#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 25 Mar 09:05:44 2011 

"""Test color conversions available in bob
"""

import argparse
import os, sys
import numpy
import unittest
import bob

def write_image(image,name):
  min = image.min()
  max = image.max()
  output_image = bob.io.Array(((image - min) * (255. / (max - min))).astype(numpy.uint8))
  
  output_image.save(name)


if __name__ == '__main__':
  
  # create command line parser object
  parser = argparse.ArgumentParser()
  # add options
  parser.add_argument('--inputimage', '-i', required=True, type=str, help='the input image to read')
  parser.add_argument('--outputimage', '-o', required=True, help='the output image to write')
  parser.add_argument('--absoutputimage', '-a', help='image name of the absolute image to write, if any')
  parser.add_argument('--part', '-p', type=str, choices=['real', 'imag', 'abs', 'phase'], default='real', help="the part of the layer to be returned")
  parser.add_argument('--layer', '-l', type=int, default=0, help='the layer of the resulting trafo image that should be written') 
  parser.add_argument('--normalize', '-n', action='store_true')
  
  # perform command line parsing
  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -o /idiap/home/mguenther/test.png -a /idiap/home/mguenther/test_abs.png".split() + sys.argv[1:])
  
  # read input image
  input_image = bob.io.Array(args.inputimage).get()
  
  # perform gwt
  gwt = bob.ip.GaborWaveletTransform()

  trafo_image = gwt.trafo_image(input_image)
  gwt.transform(input_image,trafo_image)
  
  # write output layer
  output_layer = trafo_image[args.layer]
  
  layer = {
    'real' : numpy.real(output_layer), 
    'imag' : numpy.imag(output_layer),
    'abs'  : numpy.absolute(output_layer),
    'phase': numpy.real(numpy.angle(output_layer))
  }.get(args.part)
  
  write_image(layer, args.outputimage)
  
  if 'absoutputimage' in args:
    jet_image = gwt.jet_image(input_image)
    gwt.compute_jets(input_image, jet_image, args.normalize)

    layer = jet_image[:,:,0,args.layer];
    write_image(layer, args.absoutputimage)
  

