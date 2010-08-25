#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 18:22:38 CEST 

"""This module contains the definitions of filters that are usable by
filter.py. If you want to include a filter here, follow one of the models
bellow.
"""

import torch.ip

class Filter(object):
  """Top-level class for all implemented filters"""
  pass

class Crop(Filter):
  tmp = torch.ip.ipCrop()

  doc = tmp.__doc__

  options = [ 
      (('-x',), {'type':"int", 'action':"store", 'dest':"x", 'metavar':"INT",
        'default': tmp.getIOption("x"), 
        'help':"Offset in x (defaults to %default)"}),
      (('-y',), {'type':"int", 'action':"store", 'dest':"y", 'metavar':"INT",
        'default': tmp.getIOption("y"), 
        'help':"Offset in y (defaults to %default)"}),
      (('-w','--width'), {'type': "int", 
        'action': "store", 'dest': "w", 'metavar': "INT",
        'default':tmp.getIOption("w"), 
        'help':"Width of the cropped image (defaults to %default)"}),
      (('-z','--height'), {'type': "int",
        'action': "store", 'dest': "h", 'metavar': "INT",
        'default':tmp.getIOption("h"), 
        'help':"Height of the cropped image (defaults to %default)"}),
      ]

  del tmp

  arguments = ['input', 'output']
      
  def __call__(self, options, args):
    self.filter = torch.ip.ipCrop()
    self.filter.setIOption('x', options.x)
    self.filter.setIOption('y', options.y)
    self.filter.setIOption('w', options.w)
    self.filter.setIOption('h', options.h)
    input = torch.ip.Image(1, 1, 3)
    input.load(args[0])
    if not self.filter.process(input):
      raise RuntimeError, 'Processing of "%s" has failed' % args[0]
    torch.ip.Image(self.filter.getOutput(0)).save(args[1])

class Flip(Filter):
  tmp = torch.ip.ipFlip()

  doc = tmp.__doc__

  options = [ 
      (('-v', '--vertical'), 
        {'action':"store_true", 'dest':"vertical",
        'default': tmp.getIOption("vertical"), 
        'help':"Direction of the flipping (defaults to horizontal)"}),
      ]

  del tmp

  arguments = ['input', 'output']
      
  def __call__(self, options, args):
    self.filter = torch.ip.ipCrop()
    self.filter.setBOption('vertical', options.x)
    input = torch.ip.Image(1, 1, 3)
    input.load(args[0])
    if not self.filter.process(input):
      raise RuntimeError, 'Processing of "%s" has failed' % args[0]
    torch.ip.Image(self.filter.getOutput(0)).save(args[1])

class Histo(Filter):
  tmp = torch.ip.ipHisto()

  doc = tmp.__doc__

  options = [ 
      (('-a', '--append'), 
        {'action':"store_true", 'dest':"append",
        'default': False,
        'help':"If set, I'll try to append to the output file instead of overwriting it."}),
      ]

  del tmp

  arguments = ['input-image', 'output-tensor']
      
  def __call__(self, options, args):
    self.filter = torch.ip.ipCrop()
    self.filter.setBOption('vertical', options.x)
    input = torch.ip.Image(1, 1, 3)
    input.load(args[0])
    if not self.filter.process(input):
      raise RuntimeError, 'Processing of "%s" has failed' % args[0]
    output = torch.core.TensorFile()
    otensor = self.filter.getOutput(0)
    if options.append: output.openAppend(args[1])
    else: output.openWrite(args[1], otensor)
    output.save(otensor)
    output.close()

class HistoEqual(Filter):
  tmp = torch.ip.ipHistoEqual()

  doc = tmp.__doc__

  options = []

  del tmp

  arguments = ['input', 'output']
      
  def __call__(self, options, args):
    self.filter = torch.ip.ipCrop()
    self.filter.setBOption('vertical', options.x)
    input = torch.ip.Image(1, 1, 3)
    input.load(args[0])
    if not self.filter.process(input):
      raise RuntimeError, 'Processing of "%s" has failed' % args[0]
    torch.ip.Image(self.filter.getOutput(0)).save(args[1])

# This is some black-instrospection-magic to get all filters declared in this
# submodule automatically. Don't touch it. If you want to include a new filter
# into the "filter.py" program, just declare it in this submodule and inherit
# from "Filter".
FILTERS = []
__locals__copy = dict(locals())
for k, v in __locals__copy.iteritems():
  if isinstance(v, type) and issubclass(v, Filter) and v != Filter:
    FILTERS.append(v)
del __locals__copy
