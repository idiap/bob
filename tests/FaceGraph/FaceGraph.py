#!/usr/bin/env python

import argparse
import bob
import numpy
from matplotlib import pylab
import math
import sys

def sqr(x):
  return x*x
  
class FaceGraph:

  def __init__(self,lefteye,righteye,between,along,above,below):
    """Generates a face grid graph using the given eye positions and 
    the given numebrs of nodes between, along, above, and below the eyes."""
    # shortcuts for the eye positions
    lex=lefteye[0]
    ley=lefteye[1]
    rex=righteye[0]
    rey=righteye[1]
    # compute angle between eyes
    angle=math.atan2(ley-rey,lex-rex)
    # create face grid graph
    stepx = float(lex - rex) / float(between+1)
    stepy = float(ley - rey) / float(between+1)
    xstart = float(rex) - float(along*stepx) + float(above*stepy)
    ystart = float(rey) - float(along*stepy) - float(above*stepx)
    xcount = between + 2 * (along+1)
    ycount = above + below + 1
    
    # create all grid-nodes (create columns, from left to right)
    self.nodes = list()
    for y in range(ycount) :
      for x in range(xcount) :
        xpos = round(xstart + x * stepx - y * stepy)
        ypos = round(ystart + y * stepx + x * stepy)
        self.nodes.append((xpos,ypos)) 

  def plot(self):
    for (xpos,ypos) in self.nodes:
      pylab.plot(xpos-1,ypos-1,'ro')
      
  def extract_jets(self, jet_image):
    """extracts the Gabor jets from the given jet image"""
    self.jets=[]
    for (xpos,ypos) in self.nodes:
      self.jets.append(jet_image[ypos,xpos,:,:].copy())

  def similarity_to(self,other,f):
    """computes the similarity of this graph to the given one,
    using the specified Gabor jet comparison function f"""
    sum = 0.
    for i in range(len(self.jets)):
      sum += f(self.jets[i], other.jets[i])
    sum /= len(self.jets)
    return sum
    
    

def normalize(image,eyes,neweyes,imagesize,do_normalize):
  if do_normalize:
    # compute normalization factors
    eye_pos=neweyes
    lex, ley, rex, rey = eye_pos[0], eye_pos[1], eye_pos[2], eye_pos[3]
    dist = lex - rex
    off = (lex + rex) / 2

    # normalize face image
    scaled_image = numpy.ndarray([imagesize[1], imagesize[0]], dtype=numpy.float64)
    norm = bob.ip.FaceEyesNorm(dist, imagesize[1], imagesize[0], ley, off)

    # set new eye positions  
    oep=eyes
    norm(image, scaled_image, oep[1], oep[0], oep[3], oep[2])
    return (scaled_image, neweyes)
  else:
    return (image, eyes)
  
  

if __name__ == '__main__':
  # create command line parser object
  parser = argparse.ArgumentParser()
  # add options
  parser.add_argument('--inputimage', '-i', required=True, type=str, help='the input image to read')
  parser.add_argument('--eyepositions', '-e', type=int, nargs=4, help='eye positions of the given input image')
  parser.add_argument('--normalizeface', '-x', action='store_true')
  parser.add_argument('--neweyepositions', '-n', type=int, nargs=4, default=[114,108, 54,108], help='eye positions wanted in the new image')
  parser.add_argument('--newimagesize', '-s', type=int, nargs=2, default=[168,224], help='image size of the normalized image')
  parser.add_argument('--otherimage', '-I', help='the image to compare with')
  parser.add_argument('--othereyepositions', '-E', type=int, nargs=4, help='eye positions of the other input image')
  parser.add_argument('--grid', '-g', type=int, nargs=4, default=[3,2,4,6], help='number of nodes between, aside, above, and below the eyes')
  parser.add_argument('--scales', '-S', type=int, default=5, help='number of scales of the GWT')
  
  # perform command line parsing
  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -e 223 172 129 177 -I /idiap/home/mguenther/08-Guenther-6x9-150.jpg -E 243 190 150 171".split() + sys.argv[1:])
#  args = parser.parse_args("-i /idiap/home/mguenther/10-Guenther-6x9-150.jpg -e 223 172 129 177".split() + sys.argv[1:])

  # read input image
  input_image = bob.io.Array(args.inputimage).get()
  if input_image.ndim == 3:
    gray_image = numpy.ndarray([input_image.shape[1], input_image.shape[2]], dtype=numpy.uint8)
    bob.ip.rgb_to_gray(input_image,gray_image)
  else:
    gray_image = input_image
    
  (gray_image,eye_pos) = normalize(gray_image, args.eyepositions, args.neweyepositions, args.newimagesize, args.normalizeface)

  # perform gwt
  gwt = bob.ip.GaborWaveletTransform(args.scales)
  jet_image = gwt.jet_image(gray_image)
  gwt.compute_jets(gray_image,jet_image)
  
  # create face graph
  fg = FaceGraph((eye_pos[0],eye_pos[1]), (eye_pos[2], eye_pos[3]), args.grid[0], args.grid[1], args.grid[2], args.grid[3])
  fg.extract_jets(jet_image)
  
  # show it
  pylab.figure()
  pylab.gray()
  im = pylab.imshow(gray_image)
  fg.plot()
  
  if args.otherimage:
    input_image = bob.io.Array(args.otherimage).get()
    if input_image.ndim == 3:
      other_image = numpy.ndarray([input_image.shape[1], input_image.shape[2]], dtype=numpy.uint8)
      bob.ip.rgb_to_gray(input_image,other_image)
    else:
      other_image = input_image
    
    (other_image,eye_pos) = normalize(other_image, args.othereyepositions, args.neweyepositions, args.newimagesize, args.normalizeface)
    
    fg2 = FaceGraph((eye_pos[0],eye_pos[1]), (eye_pos[2], eye_pos[3]), args.grid[0], args.grid[1], args.grid[2], args.grid[3])
    gwt.compute_jets(other_image,jet_image)
    fg2.extract_jets(jet_image)

    # show it
    pylab.figure()
    pylab.gray()
    im = pylab.imshow(other_image)
    fg2.plot()
    
    print "the similarities of the two graphs are:"
    print "Cosine Measure:" , fg.similarity_to(fg2, bob.measure.ScalarProductSimilarity())
    print "Canberra Measure:", fg.similarity_to(fg2, bob.measure.CanberraSimilarity()) 
    print "Disparity Measure:", fg.similarity_to(fg2, bob.measure.DisparitySimilarity(gwt)) 
  else:
  
    # compute the similarity of the graph to itself
    print "similarity of the face graph to itself is:", fg.similarity_to(fg, bob.measure.ScalarProductSimilarity()), fg.similarity_to(fg, bob.measure.CanberraSimilarity()), fg.similarity_to(fg, bob.measure.DisparitySimilarity(gwt)) 
    
  
  pylab.show()

