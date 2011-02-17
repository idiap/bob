#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 16 Feb 17:29:47 2011 

"""This test unit will test various features of the Torch Codec system.
"""

import os, sys
import unittest
import tempfile
import torch
import random

def get_tempfilename(prefix='torchtest_', suffix='.txt'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

# This test demonstrates how you create and add a codec that knows how to read
# data from arbitrary file types. It also tests some of the functionality the
# system is supposed to have. For this example, we will create a simple codec
# that can read and write data to a simple text file.  This (arbitrary) code
# can only digest uni-dimensional 16-bit unsigned integers. The array size is
# variable and is annotated on the file, as the first entry.

# First things first: your Codec *has* to inherit from our base class
class TextArrayCodec(torch.database.ArrayCodec):

  # For an ArrayCodec, you have to implement 5 basic methods, short of
  # which, you will get exceptions thrown at you. Here are them:
  def peek (self, filename):
    """This method briefly reads the file indicated by the first argument
    and returns a tuple (elementType, shape). Element type should be one of
    the element types defined in torch.database.ElementType. Shape is
    supposed to be a tuple indicating the array size."""

    # Well, in our case, we don't even need to look at at file, it suffices
    # to return a constant. Of course, you are allowed to do whatever you
    # want here - you have the filename and you should know what to get
    # from there. 
    #
    # An example would be reading JPEG images. You could open the file and
    # extract the image size, then return (width, height, 3) as shape and
    # uint8 as element type.
    f = open(filename, 'rt')
    entries = f.read().split() #always keep data in a single line
    return (torch.database.ElementType.uint16, (int(entries[0]),))

  def load (self, filename):
    """This method loads the whole data in the file into memory. You should
    return always a torch.database.Array object."""
    
    # We open the file given and read it
    f = open(filename, 'rt')
    entries = f.read().split() #always keep data in a single line
    f.close()
    length = int(entries[0])
    bzarray = torch.core.array.uint16_1([int(k) for k in entries[1:]], (length,))
    return torch.database.Array(bzarray)

  def save (self, filename, array):
    """This is the inverse of loading. You receive a filename and an Array,
    you put it to the file."""

    f = open(filename, 'wt')
    # In our particular case, I need to save the array in uint16, so we
    # just convert it.
    bzarray = array.cast(torch.database.ElementType.uint16)

    # It would be prudent to verify that the bzarray has the right number
    # of dimensions, but in this example we are going to trust the user.
    # bzarray.dimensions() will tell you the number of dimensions for
    # example. Please read the help for any of the blitz array bindings
    # available in Torch with help(torch.core.array.uint16_1) for example.

    f = open(filename, 'wt')
    f.write("%d " % bzarray.extent(0))
    f.write(' '.join(['%d' % bzarray[k] for k in range(bzarray.extent(0))]))
    f.close()

  def name (self):
    """This method only returns a string that indicates my unique name. By
    tradition, we like to use '.' separated namespaces. The first name
    indicates the origin framework of the codec, the second what it encodes
    or decodes (array vs. arrayset) and the third is the format name in
    which it saves. For example: "torch.array.binary" is the name of an
    array codec built-in Torch that defines our binary file formats."""

    return "example.array.text"

  def extensions (self):
    """If the user does not specifify the name of the codec to be used for
    a certain type of file, the codec is looked up in the registry using
    known file extensions. You should provide here the extensions you want
    to cover with this codec. Keep in mind that repetitions are no allowed.
    The extension should include the final '.' (dot). You are supposed to
    return a tuple in this method."""

    return ('.Xtxt',)

class CodecTest(unittest.TestCase):
  """Performs various tests for the Torch::database::*Codec* types."""
 
  def test01_CanRegisterArrayCodec(self):

    # After declaring your new codec, you need to inform Torch a new codec is
    # available, by registering it with the pertinent codec registry. Since
    # this codec can load/save arrays, it is an ArrayCodec and will be
    # registerd to the ArrayCodecRegister.
    # Please note we register an instance of a Codec.
    torch.database.ArrayCodecRegistry.addCodec(TextArrayCodec())

    # We can test the correct registration of the codec by making sure we can
    # retrieve it from the registry
    c = torch.database.ArrayCodecRegistry.getCodecByName("example.array.text")
    self.assertEqual(c.name(), "example.array.text")
    self.assertEqual(c.extensions(), ('.Xtxt',))

    # We can also retrieve all existing registered information with these
    # methods:
    self.assertEqual('example.array.text' in
        torch.database.ArrayCodecRegistry.getCodecNames(), True)
    self.assertEqual('.Xtxt' in
        torch.database.ArrayCodecRegistry.getExtensions(), True)

  def test02_CanUsePythonCodec(self):

    # This test demonstrates we can read and write using the newly registered
    # codec. For that, we will load a second database that contains special
    # entries that use this codec
    db = torch.database.Dataset('test_codec.xml')
    self.assertEqual(db[1][3].get(), 
        torch.core.array.uint16_1((4, 19, 27), (3,))) 

    # Moral for this story: once you register the codec, you can use it on your
    # databases and read it from python. Note that codecs defined in python
    # will work only on python scripts at this time. This is good for quick
    # hacking when you just want to convert data from some format to a Torch
    # native (as in "implemented in C++") format.

  def test03_Transcoding(self):

    # We can use the codecs for simple transcoding of files from one format to
    # another format. Let's take for example our "test_array_codec.txt" file in
    # the test directory. This file contains 1 array of type
    # blitz::Array<uint16_t, 1>. Let's play with it combining built-in (C++)
    # codecs and our python-based codecs and see we can go back and forth w/o
    # problems.

    text = torch.database.ArrayCodecRegistry.getCodecByName("example.array.text")
    binary = torch.database.ArrayCodecRegistry.getCodecByName("torch.array.binary")
    # Lets put the results on a temporary file we scratch later
    tmpname = get_tempfilename(suffix='.bin')
    binary.save(tmpname, text.load('test_array_codec.txt'))

    # Lets make sure the binary data saved is the same:
    bzbin = binary.load(tmpname).get()
    bztxt = text.load('test_array_codec.txt').get()
    self.assertEqual(bzbin, bztxt)

    # This trick can be done both ways!
    tmpname2 = get_tempfilename()
    text.save(tmpname2, binary.load(tmpname))

    # Lets make sure the binary data saved is the same:
    bzbin = binary.load(tmpname).get()
    bztxt = text.load(tmpname2).get()
    self.assertEqual(bzbin, bztxt)

    # And we erase both files after this
    os.unlink(tmpname)
    os.unlink(tmpname2)

    # Needless to say, you can use a similar technique to transcode a whole
    # dataset in an homogene way, by registering the codecs you want to deploy
    # and calling save() on arraysets or arrays as you see fit.

  def test04_Deregistration(self):

    # You can deregister one or many codecs using the removeCodecByName()
    # static method.
    torch.database.ArrayCodecRegistry.removeCodecByName("example.array.text")

    self.assertEqual('example.array.text' not in
        torch.database.ArrayCodecRegistry.getCodecNames(), True)
    self.assertEqual('.Xtxt' not in
        torch.database.ArrayCodecRegistry.getExtensions(), True)

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
