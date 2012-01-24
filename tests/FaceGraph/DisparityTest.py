#!/usr/bin/env python


import bob
import sys
import argparse
import numpy
from matplotlib import pylab

if __name__ == '__main__':
  # create command line parser object
  parser = argparse.ArgumentParser()
  # add options
  parser.add_argument('--inputimage', '-i', required=True, type=str, help='the input image to read')
  parser.add_argument('--testlocation', '-l', type=int, nargs=2, help='postion to test')
  parser.add_argument('--size', '-s', type=int, default=15, help='the size of the area to be scanned')
  parser.add_argument('--step', '-S', type=int, default=1, help='the stepsize of nodes in the source area')
  parser.add_argument('--scales', '-n', type=int, default=5, help='number of scales of the GWT')
  parser.add_argument('--sourceimage', '-I', help='the image that contains the source Gabor jet')
  parser.add_argument('--sourcelocation', '-L', type=int, nargs=2, help='position in the source image')

#  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -l 129 177".split() + sys.argv[1:])
#  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -l 129 177 -I /idiap/home/mguenther/08-Guenther-6x9-150.jpg -L 151 171".split() + sys.argv[1:])
  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -l 173 228".split() + sys.argv[1:])
#  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -l 173 228 -I /idiap/home/mguenther/08-Guenther-6x9-150.jpg -L 192 236".split() + sys.argv[1:])

  # read input image
  input_image = bob.io.Array(args.inputimage).get()
  if input_image.ndim == 3:
    gray_image = numpy.ndarray([input_image.shape[1], input_image.shape[2]], dtype=numpy.uint8)
    bob.ip.rgb_to_gray(input_image,gray_image)
  else:
    gray_image = input_image

  # perform gwt
  gwt = bob.ip.GaborWaveletTransform(number_of_scales=args.scales)
  jet_image = gwt.jet_image(gray_image)
  gwt.compute_jets(gray_image,jet_image)

  # get reference Gabor jet
  if args.sourceimage:
    print "reading second image"
    source_image = bob.io.Array(args.sourceimage).get()
    if source_image.ndim == 3:
      gray_image2 = numpy.ndarray([source_image.shape[1], source_image.shape[2]], dtype=numpy.uint8)
      bob.ip.rgb_to_gray(source_image,gray_image2)
    else:
      gray_image2 = source_image
      
    jet_image2 = gwt.jet_image(gray_image2)
    gwt.compute_jets(gray_image2,jet_image2)
    
    reference_jet = jet_image2[args.sourcelocation[1], args.sourcelocation[0], :, :]
#    print reference_jet
  else:
    reference_jet = jet_image[args.testlocation[1], args.testlocation[0], :, :]
    
  # compute disparities and show them
  pylab.figure()
  pylab.gray()
  im = pylab.imshow(gray_image)
  cm = pylab.cm.get_cmap('jet')
  
  disp = bob.measure.DisparitySimilarity(gwt)
  for x in range(args.testlocation[0] - args.size, args.testlocation[0] + args.size + 1, args.step):
    for y in range(args.testlocation[1] - args.size, args.testlocation[1] + args.size + 1, args.step):
#  for x in range(args.testlocation[0], args.testlocation[0]+1):
#    for y in range(args.testlocation[1],args.testlocation[1]+1):
      sim = disp(reference_jet, jet_image[y,x,:,:])
#      print jet_image[y,x,:,:]
#      print x,y,disp.disparity(), sim
      pylab.arrow(x,y,disp.disparity()[0],disp.disparity()[1], color=cm(sim))
      

  pylab.xlim(args.testlocation[0] - 3*args.size/2, args.testlocation[0] + 3*args.size/2)
  pylab.ylim(args.testlocation[1] + 3*args.size/2, args.testlocation[1] - 3*args.size/2)
  pylab.show()
