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
# variable and is annotated on the file, as the first entry, together with the
# number of samples. The following lines are the arrays, one per line.

# First things first: your Codec *has* to inherit from our base class
class TextArraysetCodec(torch.database.ArraysetCodec):

  # For an ArraysetCodec, you have to implement 7 basic methods, short of
  # which, you will get exceptions thrown at you. Here are them:

  def peek (self, filename):
    """This method briefly reads the file indicated by the first argument
    and returns a tuple (elementType, shape, samples). Element type should be
    one of the element types defined in torch.database.ElementType. Shape is
    supposed to be a tuple indicating the array size."""

    # Well, in our case, we don't even need to look at at file, it suffices
    # to return a constant. Of course, you are allowed to do whatever you
    # want here - you have the filename and you should know what to get
    # from there. 
    f = open(filename, 'rt')
    info = f.readline().split()
    f.close()
    return (torch.database.ElementType.uint16, (int(info[0]),), int(info[1]))

  def load (self, *args):
    """This method is overloaded and should present 2 distinct behaviors: 1) if
    you pass 1 argument (len(args) == 1), it should be the filename and we
    should completely load the file and return an Arrayset; 2) if you pass 2
    arguments (len(args) == 2), it should pass just pick one array in the file,
    defined by args[1] index."""

    if len(args) == 1: #load all
      f = open(args[0], 'rt')
      info = f.readline().split() #always keep data in a single line
      data = [k.split() for k in f]
      length = int(info[0])
      retval = torch.database.Arrayset()
      for d in data:
        bzarray = torch.core.array.uint16_1([int(k) for k in d], (length,))
        retval.append(bzarray)
      return retval

    elif len(args) == 2: #load a specific array only
      f = open(args[0], 'rt')
      info = f.readline().split() #always keep data in a single line
      data = [k.split() for k in f][args[1]-1]
      length = int(info[0])
      bzarray = torch.core.array.uint16_1([int(k) for k in data], (length,))
      return torch.database.Array(bzarray)

    #to make it nice, we throw a TypeError indicating the wrong len(args)
    raise TypeError, "load() takes either 1 or 2 extra arguments (%d given)""" % len(args)

  def save (self, filename, arrayset):
    """This is the inverse of loading. You receive a filename and an Array,
    you put it to the file."""

    f = open(filename, 'wt')
    f.write('%d %d\n' % (arrayset.shape[0], len(arrayset)))
    # In our particular case, I need to save the array in uint16, so we
    # just convert it.
    for k in arrayset.ids():
      bzarray = arrayset[k].cast(torch.database.ElementType.uint16)
      f.write(' '.join(['%d' % bzarray[k] for k in range(bzarray.extent(0))]))
      f.write('\n')
    f.close()

  def append (self, filename, array):
    """In here we append a single array to an existing file"""

    # It would be prudent to check if this array satisfies the file condition.
    # To do that we could call peek() and then cross-check. For this
    # simplistic example we will only write and pretend to "trust" the user.

    f = open(filename, 'w+') #append
    bzarray = array.cast(torch.database.ElementType.uint16)
    f.write(' '.join(['%d' % bzarray[k] for k in range(bzarray.extent(0))]))
    f.write('\n')
    f.close()

  def name (self):
    """This method only returns a string that indicates my unique name. By
    tradition, we like to use '.' separated namespaces. The first name
    indicates the origin framework of the codec, the second what it encodes
    or decodes (array vs. arrayset) and the third is the format name in
    which it saves. For example: "torch.array.binary" is the name of an
    array codec built-in Torch that defines our binary file formats."""

    return "example.arrayset.text"

  def extensions (self):
    """If the user does not specifify the name of the codec to be used for
    a certain type of file, the codec is looked up in the registry using
    known file extensions. You should provide here the extensions you want
    to cover with this codec. Keep in mind that repetitions are no allowed.
    The extension should include the final '.' (dot). You are supposed to
    return a tuple in this method."""

    return ('.XXtxt',)

class ArraysetCodecTest(unittest.TestCase):
  """Performs various tests for the Torch::database::*Codec* types."""
 
  def test01_CanRegisterArraysetCodec(self):

    # This is the same as test01, but with Array codecs, so bare with us
    torch.database.ArraysetCodecRegistry.addCodec(TextArraysetCodec())

    # We can test the correct registration of the codec by making sure we can
    # retrieve it from the registry
    c = torch.database.ArraysetCodecRegistry.getCodecByName("example.arrayset.text")
    self.assertEqual(c.name(), "example.arrayset.text")
    self.assertEqual(c.extensions(), ('.XXtxt',))

    # We can also retrieve all existing registered information with these
    # methods:
    self.assertEqual('example.arrayset.text' in
        torch.database.ArraysetCodecRegistry.getCodecNames(), True)
    self.assertEqual('.XXtxt' in
        torch.database.ArraysetCodecRegistry.getExtensions(), True)

  def test02_CanUsePythonArraysetCodec(self):

    # This test demonstrates we can read and write using the newly registered
    # arrayset codec, in a transparent manner. We load a database that refers
    # to an external arrayset encoded using our new technique. This is
    # equivalent to what was done in test02 in the ArrayCodec tests.
    db = torch.database.Dataset('test_set_codec.xml')
    self.assertEqual(db[1][1].get(), 
        torch.core.array.uint16_1((4, 19, 27), (3,))) 
    self.assertEqual(db[1][4].get(), 
        torch.core.array.uint16_1((10, 15, 12), (3,))) 
    self.assertEqual(db[1][5].get(), 
        torch.core.array.uint16_1((99, 122, 1020), (3,))) 

    # We can also load the whole arrayset in one shot
    arrayset = db[1]
    arrayset.load() #triggers the load(filename) callback in the codec
    self.assertEqual(len(arrayset), 5)
    self.assertEqual(arrayset.shape, (3,))
    self.assertEqual(arrayset[3], db[1][3])

  def test03_Transcoding(self):

    # We can use the codecs for simple transcoding of files from one format to
    # another format. Let's take for example our "test_arrayset_codec.txt" file
    # in the test directory. This file contains 5 arrays of type
    # blitz::Array<uint16_t, 1> We can load using a codec and just write using
    # the other one. Let's play combining Torch built-in (C++) codecs and show
    # you can go back and forth seamlessly.

    text = torch.database.ArraysetCodecRegistry.getCodecByName("example.arrayset.text")
    binary = torch.database.ArraysetCodecRegistry.getCodecByName("torch.arrayset.binary")
    # Lets put the results on a temporary file we scratch later
    tmpname = get_tempfilename(suffix='.bin')
    binary.save(tmpname, text.load('test_arrayset_codec.txt'))

    # Lets make sure the binary data saved is the same:
    binset = binary.load(tmpname)
    txtset = text.load('test_arrayset_codec.txt')
    self.assertEqual(binset, txtset)

    # Notice: Arrayset comparisons, as described in their help message, do not
    # compare roles, but just shape, length and numerical equality of the
    # arrays within the Arrayset. If you want to compare the role, you should
    # do that additionaly to using set1 == set2.

    # This transcoding trick can be done both ways, of course!
    tmpname2 = get_tempfilename()
    text.save(tmpname2, binary.load(tmpname))

    # Lets make sure the binary data saved is the same:
    binset = binary.load(tmpname)
    txtset = text.load(tmpname2)
    self.assertEqual(binset, txtset)

    # And we erase both files after this
    os.unlink(tmpname)
    os.unlink(tmpname2)

    # Needless to say, you can use a similar technique to transcode a whole
    # dataset in an homogene way, by registering the codecs you want to deploy
    # and calling save() on arraysets or arrays as you see fit.

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
