#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  8 Mar 16:54:24 2011 

"""Converts .flo files into Torch .bin
"""

import os, sys, struct
import torch

def getf(f):
  return struct.unpack("f", f.read(4))[0]

def geti(f):
  return struct.unpack("i", f.read(4))[0]

class FloArrayCodec(torch.database.ArrayCodec):

  def peek (self, filename):
    """This method briefly reads the file indicated by the first argument
    and returns a tuple (elementType, shape). Element type should be one of
    the element types defined in torch.database.ElementType. Shape is
    supposed to be a tuple indicating the array size."""
    TAG_FLOAT = 202021.25;
    f = open(filename, 'rb')
    tag = getf(f)
    if tag != TAG_FLOAT:
      raise RuntimeError, \
        "Tag value does not check %.2f != %.2f" % (TAG_FLOAT, tag)
    width = geti(f)
    height = geti(f)
    return (torch.database.ElementType.float32, (height, width, 2))

  def load (self, filename):
    """This method loads the whole data in the file into memory. You should
    return always a torch.database.Array object."""
    
    TAG_FLOAT = 202021.25;
    f = open(filename, 'rb')
    tag = getf(f)
    if tag != TAG_FLOAT:
      raise RuntimeError, \
        "Tag value does not check %.2f != %.2f" % (TAG_FLOAT, tag)
    width = geti(f)
    height = geti(f)
    bzarray = torch.core.array.float32_3(height, width, 2)
    for j in range(width):
      for k in range(2): #u and then v
        for i in range(height):
          bzarray[i,j,k] = getf(f)
    return torch.database.Array(bzarray)

  def save (self, filename, array):
    """This is the inverse of loading. You receive a filename and an Array,
    you put it to the file."""

    raise RuntimeError, "Flow array saving not implemented"

  def name (self):
    """This method only returns a string that indicates my unique name. By
    tradition, we like to use '.' separated namespaces. The first name
    indicates the origin framework of the codec, the second what it encodes
    or decodes (array vs. arrayset) and the third is the format name in
    which it saves. For example: "torch.array.binary" is the name of an
    array codec built-in Torch that defines our binary file formats."""

    return "flow.array.binary"

  def extensions (self):
    """If the user does not specifify the name of the codec to be used for
    a certain type of file, the codec is looked up in the registry using
    known file extensions. You should provide here the extensions you want
    to cover with this codec. Keep in mind that repetitions are no allowed.
    The extension should include the final '.' (dot). You are supposed to
    return a tuple in this method."""

    return ('.flo',)

if __name__ == "__main__":
  if len(sys.argv) == 1:
    print "Transcodes from .flo to any other Torch format"
    print "usage: %s from-file.flo <to-file>" % os.path.basename(sys.argv[0])
    sys.exit(1)
  torch.database.ArrayCodecRegistry.addCodec(FloArrayCodec())
  torch.database.array_transcode(sys.argv[1], sys.argv[2])
