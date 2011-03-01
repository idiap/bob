#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 28 Feb 11:55:47 2011 

"""A set of tests for PathList objects.
"""

import os, sys
import torch
import unittest

class PathListTest(unittest.TestCase):
  """Performs various tests for the Torch::database::PathList type"""

  def test01_create(self):

    # This test examplifies how you can create a PathList object in python and
    # how to append stuff to it. 
    pl = torch.database.PathList()

    # Appending adds a new path to the end of the internal search list.
    pl.append('/input/path/foobar1')

    # You can access all internal paths, on their search order using the
    # paths() method.
    self.assertEqual(pl.paths(), ('/input/path/foobar1',))

    # Prepending adds a new path to the begin of the internal search list.
    pl.prepend('/input/path/pre2')
    self.assertEqual(pl.paths(), ('/input/path/pre2', '/input/path/foobar1'))

    # Please note that if you append (or prepend) an existing path inside the
    # PathList, that path is first removed and then prepended/appended.
    pl.append('/input/path/pre2')
    self.assertEqual(pl.paths(), ('/input/path/foobar1', '/input/path/pre2'))

  def test02_locate(self):

    # You can create an empty PathList, but to make to make this example a
    # little bit more fun, we are going to use the PathList to implement the
    # "executable file lookup" that is implemented in UNIX shell. In the shell,
    # this is driven by the environment variable PATH. Everytime you need a
    # utility, such as 'ls', the shell looks up the first path in that list,
    # which contains that executable program. 

    # You can create a path list starting from a UNIX path in the format
    # "path1:path2:path3". Here is an example:

    pl = torch.database.PathList(os.environ['PATH'])

    # Now we should locate the program 'ls'. Normally, it should point to
    # '/bin/ls'
    self.assertEqual(pl.locate('ls'), '/bin/ls')

    # And, if we remove the '/bin' directory from the search path, 'ls' should
    # not be found anymore...
    pl.remove('/bin')
    self.assertEqual(pl.locate('ls'), '') #N.B.: empty == not found

  def test03_reduce(self):

    # One of the functionalities provided by the PathList class is the ability
    # to reduce a path and to make it relative. It does this by comparing an
    # absolute path with the values of internal paths and spitting out just the
    # relative that that could be resolved correctly if locate() is called.

    # Let's go back to the PATH example and the location of 'ls'
    
    pl = torch.database.PathList(os.environ['PATH'])
    absolute = pl.locate('ls') #absolute == '/bin/ls'
    relative = pl.reduce('ls') #relative == 'ls'

    self.assertEqual(relative, 'ls')

  def test04_resolution(self):

    # The paths in the PathList are not resolved until they are neded. Suppose
    # for example I put '.' in the PathList, then if I switch to '/bin', I
    # should be able to locate 'ls'. The use of locate() or reduce() will use
    # always absolute paths
    pl = torch.database.PathList('.')

    # Make sure '.' is in there
    self.assertTrue('.' in pl)

    os.chdir('/bin')
    
    # Now I can resolve ls, since I have '.' in PathList and ls is in /bin,
    # please note that the locate() method always returns full paths, even if
    # it contains relative paths inside.
    self.assertEqual(pl.locate('ls'), '/bin/ls')

    # I can also reduce the path while in there
    self.assertEqual(pl.reduce('/bin/ls'), 'ls')

    # Using relative paths should get you correctly resolved
    self.assertEqual(pl.locate('./ls'), '/bin/ls')
    self.assertEqual(pl.locate('../bin/ls'), '/bin/ls')
    
    # If you use '..', we ignore anything that goes beyond root
    self.assertEqual(pl.locate('../../../../../bin/ls'), '/bin/ls')

    # You can use combinations of '.' and '..' as you wish
    self.assertEqual(pl.locate('.././././.././../bin/ls'), '/bin/ls')

    # The same as for reducing
    self.assertEqual(pl.reduce('/bin/../bin/ls'), 'ls')
    self.assertEqual(pl.reduce('/bin/.././../bin/ls'), 'ls')

    # Reduction returns the input when it cannot resolve the name
    self.assertEqual(pl.reduce('/does/not/exist'), '/does/not/exist')
    self.assertEqual(pl.reduce('../test/blaBla'), '../test/blaBla')
    self.assertEqual(pl.reduce('../../../test/blaBla'), '../../../test/blaBla')

    # If I move to another directory that does not contain 'ls', the
    # resolution stops working. This works as designed. It is important that
    # locate() always returns an existing path, while reduce() leaves alone
    # paths it cannot resolve for some reason.
    os.chdir('/')
    self.assertEqual(pl.locate('ls'), '')

    # localisation will work again if I specify a bit more on the path
    self.assertEqual(pl.locate('bin/ls'), '/bin/ls')
    self.assertEqual(pl.locate('bin/../bin/.././bin/ls'), '/bin/ls')

    # reduction will also work as expected, even if that means only removing
    # the front slash
    self.assertEqual(pl.reduce('/bin/ls'), 'bin/ls')

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

