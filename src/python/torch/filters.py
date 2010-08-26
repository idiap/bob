#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 18:22:38 CEST 

"""This module contains the definitions of filters that are usable by
filter.py. If you want to include a filter here, follow one of the models
bellow.
"""

import os
import torch

def map_variable_types(v):
  """Returns the standard python types for variables defined"""
  if v.type == torch.core.VariableType.Int: return 'int'
  elif v.type in (torch.core.VariableType.Float, torch.core.VariableType.Double): return 'float'
  elif v.type == torch.core.VariableType.Bool: return 'bool'
  elif v.type == torch.core.VariableType.String: return 'str'
  raise RuntimeError, 'Unsupported mapping from type %s' % v.type

def generate_option_dict(o, var):
  """Generates the dictionary required for option parsing, from a specific
  variable in an object"""
  v = [k for k in o.variables() if k.name == var]
  if len(v) != 1: 
    raise RuntimeError, 'Object does not contain Variable %s' % var
  v = v[0]
  t = map_variable_types(v)
  action = 'store'
  metavar = t.upper()
  if t == 'bool':
    if o.getBOption(var):
      action = 'store_false'
      default = True
    else:
      action = 'store_true'
      default = False
    metavar = None
  elif t == 'str':
    default = o.getSOption(var)
  elif t == 'int':
    default = o.getIOption(var)
  if v.type == torch.core.VariableType.Float:
    default = o.getFOption(var)
  if v.type == torch.core.VariableType.Double:
    default = o.getDOption(var)

  if metavar:
    return {'type': t, 'action': action, 'dest': var, 'metavar': metavar,
            'default': default, 'help': v.help + ' (defaults to %default)'}
  return {'action': action, 'dest': var,
          'default': default, 'help': v.help + ' (defaults to %default)'}

def customize_filter(filter, options):
  for k, v in filter.variable_dict().iteritems():
    if hasattr(options, k):
      if v.type == torch.core.VariableType.Bool:
        filter.setBOption(k, getattr(options, k))
      elif v.type == torch.core.VariableType.Int:
        filter.setIOption(k, getattr(options, k))
      elif v.type == torch.core.VariableType.Float:
        filter.setFOption(k, getattr(options, k))
      elif v.type == torch.core.VariableType.Double:
        filter.setDOption(k, getattr(options, k))
      elif v.type == torch.core.VariableType.String:
        filter.setSOption(k, getattr(options, k))
      else:
        raise RuntimeError, 'I cannot customize option of type %s' % v.type

def apply_filter_from_instance(filter, input, output, planes):
  if not os.path.exists(input):
    raise RuntimeError, 'I cannot read input file "%s"' % input
  i = torch.ip.Image(1, 1, planes)
  i.load(input)
  if not filter.process(i):
    raise RuntimeError, 'Processing of "%s" has failed' % input
  if not filter.getNOutputs() == 1:
    raise RuntimeError, 'Filter "%s" returned more than 1 output?' % filter
  torch.ip.Image(filter.getOutput(0)).save(output)

def apply_image_filter(cls, options, input, output, planes=3):
  filter = cls()
  customize_filter(filter, options)
  apply_filter_from_instance(filter, input, output, planes)

def apply_image_processor(cls, options, input, output, planes=3):
  filter = cls()
  customize_filter(filter, options)
  if not os.path.exists(input):
    raise RuntimeError, 'I cannot read input file "%s"' % input
  i = torch.ip.Image(1, 1, planes)
  i.load(input)
  if not filter.process(i):
    raise RuntimeError, 'Processing of "%s" has failed' % input
  o = torch.core.TensorFile()
  otensor = filter.getOutput(0)
  if options.append: o.openAppend(output)
  else: o.openWrite(output, otensor)
  for n in range(filter.getNOutputs()): o.save(filter.getOutput(n))
  o.close()

APPEND_OPTION = (('-a', '--append'), 
    {'action': "store_true", 'dest': "append", 'default': False,
     'help': "If set, I'll try to append to the output file instead of overwriting it."})

class Filter(object):
  """Top-level class for all implemented filters"""
  pass

class Crop(Filter):
  tmp = torch.ip.ipCrop()

  doc = tmp.__doc__

  options = [ 
      (('-x',), generate_option_dict(tmp, 'x')),
      (('-y',), generate_option_dict(tmp, 'y')),
      (('-w', '--width'), generate_option_dict(tmp, 'w')),
      (('-z', '--height'), generate_option_dict(tmp, 'h')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    if options.w == 0 or options.h == 0:
      raise RuntimeError, 'I cannot crop an image to have zero dimensions, please revise your options!'
    apply_image_filter(torch.ip.ipCrop, options, args[0], args[1], planes=3)

class Flip(Filter):
  tmp = torch.ip.ipFlip()

  doc = tmp.__doc__

  options = [ 
      (('-v', '--vertical'), generate_option_dict(tmp, 'vertical')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipFlip, options, args[0], args[1], planes=3)

class Histo(Filter):
  tmp = torch.ip.ipHisto()

  doc = tmp.__doc__

  options = [APPEND_OPTION] 

  del tmp

  arguments = ['input-image', 'output-tensor']
      
  def __call__(self, options, args):
    apply_image_processor(torch.ip.ipHisto, options, 
        args[0], args[1], planes=3)

class HistoEqual(Filter):
  tmp = torch.ip.ipHistoEqual()

  doc = tmp.__doc__

  options = []

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipHistoEqual, options, 
        args[0], args[1], planes=1)

class Integral(Filter):
  tmp = torch.ip.ipIntegral()

  doc = tmp.__doc__

  options = [APPEND_OPTION]

  del tmp

  arguments = ['input-image', 'output-tensor']
      
  def __call__(self, options, args):
    apply_image_processor(torch.ip.ipIntegral, options, 
        args[0], args[1], planes=3)

class lbp8R(Filter):
  tmp = torch.ip.ipLBP8R()

  doc = tmp.__doc__

  options = [ 
      (('-a','--to-average'), generate_option_dict(tmp, 'ToAverage')),
      (('-b','--average-bit'), generate_option_dict(tmp, 'AddAvgBit')),
      (('-u','--uniform'), generate_option_dict(tmp, 'Uniform')),
      (('-i','--rotation-invariant'), generate_option_dict(tmp, 'RotInvariant')),
      (('-r', '--radius'),  {'action': "store", 'dest': "radius", 
        'default': 1, 
        'help': "Sets the radius, in pixels, of the LBP operator (defaults to %default)"}),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    #the LBP implementation in torch requires special handling.
    filter = torch.ip.ipLBP8R(options.radius)
    customize_filter(filter, options)
    if not os.path.exists(args[0]):
      raise RuntimeError, 'I cannot read input file "%s"' % args[0]
    i = torch.ip.Image(1, 1, 3)
    i.load(args[0])
    o = filter.batch(i)
    if not o:
      raise RuntimeError, 'Processing of "%s" has failed' % args[0]
    o.save(args[1])

class lbp4R(Filter):
  tmp = torch.ip.ipLBP4R()

  doc = tmp.__doc__

  options = [ 
      (('-a','--to-average'), generate_option_dict(tmp, 'ToAverage')),
      (('-b','--average-bit'), generate_option_dict(tmp, 'AddAvgBit')),
      (('-u','--uniform'), generate_option_dict(tmp, 'Uniform')),
      (('-i','--rotation-invariant'), generate_option_dict(tmp, 'RotInvariant')),
      (('-r', '--radius'),  {'action': "store", 'dest': "radius", 
        'default': 1, 
        'help': "Sets the radius, in pixels, of the LBP operator (defaults to %default)"}),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    #the LBP implementation in torch requires special handling.
    filter = torch.ip.ipLBP4R(options.radius)
    customize_filter(filter, options)
    if not os.path.exists(args[0]):
      raise RuntimeError, 'I cannot read input file "%s"' % args[0]
    i = torch.ip.Image(1, 1, 3)
    i.load(args[0])
    o = filter.batch(i)
    if not o:
      raise RuntimeError, 'Processing of "%s" has failed' % args[0]
    o.save(args[1])

class MSRSQIGaussian(Filter):
  tmp = torch.ip.ipMSRSQIGaussian()

  doc = tmp.__doc__

  options = [ 
      (('-x','--radius-x'), generate_option_dict(tmp, 'RadiusX')),
      (('-y','--radius-y'), generate_option_dict(tmp, 'RadiusY')),
      (('-s','--sigma'), generate_option_dict(tmp, 'Sigma')),
      (('-w', '--weighted'), generate_option_dict(tmp, 'Weighed')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipMSRSQIGaussian, options, args[0], args[1], 
        planes=3) 

class MultiscaleRetinex(Filter):
  tmp = torch.ip.ipMultiscaleRetinex()

  doc = tmp.__doc__

  options = [ 
      (('-n','--scales'), generate_option_dict(tmp, 's_nb')),
      (('-m','--min'), generate_option_dict(tmp, 's_min')),
      (('-t','--step'), generate_option_dict(tmp, 's_step')),
      (('-s', '--sigma'), generate_option_dict(tmp, 'sigma')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipMultiscaleRetinex, options, args[0], args[1], planes=1)

class Relaxation(Filter):
  tmp = torch.ip.ipRelaxation()

  doc = tmp.__doc__

  options = [ 
      (('-t','--type'), generate_option_dict(tmp, 'type')),
      (('-s','--steps'), generate_option_dict(tmp, 'steps')),
      (('-l','--lambda'), generate_option_dict(tmp, 'lambda')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipRelaxation, options, args[0], args[1], planes=1)

class Rotate(Filter):
  tmp = torch.ip.ipRotate()

  doc = tmp.__doc__

  options = [ 
      (('-x','--center-x'), generate_option_dict(tmp, 'centerx')),
      (('-y','--center-y'), generate_option_dict(tmp, 'centery')),
      (('-a','--angle'), generate_option_dict(tmp, 'angle')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipRotate, options, args[0], args[1], planes=3)

class Rotate(Filter):
  tmp = torch.ip.ipScaleYX()

  doc = tmp.__doc__

  options = [ 
      (('-w','--width'), generate_option_dict(tmp, 'width')),
      (('-t','--height'), generate_option_dict(tmp, 'height')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipScaleYX, options, args[0], args[1], planes=3)

class SelfQuotient(Filter):
  tmp = torch.ip.ipSelfQuotientImage()

  doc = tmp.__doc__

  options = [ 
      (('-s','--scales'), generate_option_dict(tmp, 's_nb')),
      (('-m','--min'), generate_option_dict(tmp, 's_min')),
      (('-t','--step'), generate_option_dict(tmp, 's_step')),
      (('-g','--sigma'), generate_option_dict(tmp, 'Sigma')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipSelfQuotientImage, options, args[0], args[1], planes=3)

class Shift(Filter):
  tmp = torch.ip.ipShift()

  doc = tmp.__doc__

  options = [ 
      (('-x','--shift-x'), generate_option_dict(tmp, 'shiftx')),
      (('-y','--shift-y'), generate_option_dict(tmp, 'shifty')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipShift, options, args[0], args[1], planes=3)

class SmoothGaussian(Filter):
  tmp = torch.ip.ipSmoothGaussian()

  doc = tmp.__doc__

  options = [ 
      (('-x','--radius-x'), generate_option_dict(tmp, 'RadiusX')),
      (('-y','--radius-y'), generate_option_dict(tmp, 'RadiusY')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipSmoothGaussian, options, args[0], args[1], planes=3)

class Sobel(Filter):
  tmp = torch.ip.ipSobel()

  doc = tmp.__doc__

  options = []

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipSobel, options, args[0], args[1], planes=3)

class TanTriggs(Filter):
  tmp = torch.ip.ipTanTriggs()

  doc = tmp.__doc__

  options = [ 
      (('-s','--step'), generate_option_dict(tmp, 's_step')),
      (('-g','--gamma'), generate_option_dict(tmp, 'gamma')),
      (('-z','--sigma0'), generate_option_dict(tmp, 'sigma0')),
      (('-y','--sigma1'), generate_option_dict(tmp, 'sigma1')),
      (('-t','--size'), generate_option_dict(tmp, 'size')),
      (('-q','--threshold'), generate_option_dict(tmp, 'threshold')),
      (('-a','--alpha'), generate_option_dict(tmp, 'alpha')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipTanTriggs, options, args[0], args[1], planes=3)

class Vcycle(Filter):
  tmp = torch.ip.ipVcycle()

  doc = tmp.__doc__

  options = [ 
      (('-l','--lambda'), generate_option_dict(tmp, 'lambda')),
      (('-g','--grids'), generate_option_dict(tmp, 'n_grids')),
      (('-t','--type'), generate_option_dict(tmp, 'type')),
      ]

  del tmp

  arguments = ['input-image', 'output-image']
      
  def __call__(self, options, args):
    apply_image_filter(torch.ip.ipVcycle, options, args[0], args[1], planes=3)

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
