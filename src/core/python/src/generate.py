#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 23 Nov 07:43:55 2010 

"""Runs the template generation program
"""

import os, sys

template = """/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @brief blitz::Array<%(type)s,%(dim)d> to and from python converters
 */
#include "core/python/array.h"
%(macro)s(%(type)s, %(dim)d, %(typestr)s, %(fname)s)
"""

# The list of c++ types that will be generated
# Hint: start with most problematic types that allow you to discover problems
# early in the compilation process!
types = (
    { #fast compilation, show lots of problems if they occur
      'typestr': 'bool',
      'type': 'bool',
      'macro': 'declare_bool_array',
    },
    { #blitz sometimes does not include functions to operate on chars
      'typestr': 'int8',
      'type': 'int8_t',
      'macro': 'declare_integer_array',
    },
    { #idem
      'typestr': 'uint8',
      'type': 'uint8_t',
      'macro': 'declare_unsigned_array',
    },
    { #in 32-bit machines, this is normally long long int, in 64-bit, a long
      #int is a 64-bit number.
      'typestr': 'int64',
      'type': 'int64_t',
      'macro': 'declare_integer_array',
    },
    {
      'typestr': 'uint64',
      'type': 'uint64_t',
      'macro': 'declare_unsigned_array',
    },
    { #first occurence of float compilation
      'typestr': 'float32',
      'type': 'float',
      'macro': 'declare_float_array',
    },
    { #first occurence of complex compilation
      'typestr': 'complex64',
      'type': 'std::complex<float>', 
      'macro': 'declare_complex_array',
    },
    {
      'typestr': 'int16',
      'type': 'int16_t',
      'macro': 'declare_integer_array',
    },
    {
      'typestr': 'uint16',
      'type': 'uint16_t',
      'macro': 'declare_unsigned_array',
    },
    {
      'typestr': 'float64',
      'type': 'double',
      'macro': 'declare_float_array',
    },
    {
      'typestr': 'int32',
      'type': 'int32_t',
      'macro': 'declare_integer_array',
    },
    {
      'typestr': 'uint32',
      'type': 'uint32_t',
      'macro': 'declare_unsigned_array',
    },
    {
      'typestr': 'float128', 
      'type': 'long double',
      'macro': 'declare_float_array',
    },
    {
      'typestr': 'complex128',
      'type': 'std::complex<double>', 
      'macro': 'declare_complex_array',
    },
    {
      'typestr': 'complex256', 
      'type': 'std::complex<long double>', 
      'macro': 'declare_complex_array',
    },
)

# How many dimensions to cover
dimensions = (1, 2, 3, 4)

# Output filename template
output_template = 'array_%(typestr)s_%(dim)d.cc'

# The function names
fname_template = 'bind_core_array_%(typestr)s_%(dim)d'

def main():
  for dim in dimensions:
    for t in types: 
      d = dict(t)
      d['dim'] = dim
      output_filename = output_template % d
      d['fname'] = fname_template % d
      f = open(output_filename, 'wt')
      f.write(template % d)
      f.close()
      print "Created %s" % output_filename

  # Prints the output for the main function:
  print "\nPut this in your main():"
  print "// START: %s" % os.path.basename(sys.argv[0]) 
  for dim in dimensions:
    for t in types: 
      d = dict(t)
      d['dim'] = dim
      d['fname'] = fname_template % d
      print 'void %(fname)s();' % d 
  print "// END: %s" % os.path.basename(sys.argv[0]) 
  print "// START: %s" % os.path.basename(sys.argv[0]) 
  for dim in dimensions:
    for t in types: 
      d = dict(t)
      d['dim'] = dim
      d['fname'] = fname_template % d
      print '%(fname)s();' % d 
  print "// END: %s" % os.path.basename(sys.argv[0]) 

  # Prints out the output for the cmake file:
  print "\nPut this in your CMakeLists.txt:"
  for dim in dimensions:
    for t in types: 
      d = dict(t)
      d['dim'] = dim
      output_filename = output_template % d
      print '   "python/src/%s"' % output_filename

if __name__ == '__main__':
  main()
