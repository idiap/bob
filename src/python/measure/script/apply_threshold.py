#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 24 May 2011 10:28:12 CEST

"""This script applies a threshold to score file and reports error rates
"""

__epilog__ = """
Examples:

  1. Standard usage

     $ %(prog)s --scores=my-scores.txt --threshold=0.5

Note: 

This is just an example program. It is not meant to be perfect or complete,
just to give you the basis to develop your own scripts. In order to tweak more
options, just copy this file to your directory and modify it to fit your needs.
You can easily copy this script like this:

  $ cp `which %(prog)s` .
  $ vim %(prog)s
"""

import sys, os, torch

def apthres(neg, pos, thres):
  """Prints a single output line that contains all info for the threshold"""

  far, frr = torch.measure.farfrr(neg, pos, thres)
  hter = (far + frr)/2.0

  ni = neg.extent(0) #number of impostors
  fa = int(round(far*ni)) #number of false accepts
  nc = pos.extent(0) #number of clients
  fr = int(round(frr*nc)) #number of false rejects

  print("FAR : %.3f%% (%d/%d)" % (100*far, fa, ni))
  print("FRR : %.3f%% (%d/%d)" % (100*frr, fr, nc))
  print("HTER: %.3f%%" % (100*hter,))

def get_options():
  """Parse the program options"""

  import optparse
  
  class MyParser(optparse.OptionParser):
    def format_epilog(self, formatter):
      return self.epilog
    def format_description(self, formatter):
      return self.description

  usage = 'usage: %s [arguments]' % os.path.basename(sys.argv[0])

  parser = MyParser(usage=usage, 
      description=(__doc__ % {'prog': os.path.basename(sys.argv[0])}),
      epilog=(__epilog__ % {'prog': os.path.basename(sys.argv[0])}))

  parser.add_option('-s', '--scores', dest="ifile", default=None,
      help="Name of the file containing the scores (defaults to %default)",
      metavar="FILE")
  parser.add_option('-t', '--threshold', dest='thres', default=None,
      type='float', help="The threshold value to apply", metavar="FLOAT")
  parser.add_option('-p', '--parser', dest="parser", default="4column",
      help="Name of a known parser or of a python-importable function that can parse your input files and return a tuple (negatives, positives) as blitz 1-D arrays of 64-bit floats. Consult the API of torch.measure.load.split_four_column() for details", metavar="NAME.FUNCTION")
  
  # This option is not normally shown to the user...
  parser.add_option("--self-test",
      action="store_true", dest="test", default=False,
      help=optparse.SUPPRESS_HELP)
      #help="if set, runs an internal verification test and erases any output")

  options, args = parser.parse_args()

  if options.test:
    # then we go into test mode, all input is preset
    packdir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    options.ifile = os.path.join(packdir, 'test', 'data', 'test-4col.txt')
    options.thres = 0.0

  if options.ifile is None:
    parser.error("you should give an input score set with --scores")

  if options.thres is None:
    parser.error("you should give a threshold value with --threshold")

  #parse the score-parser
  if options.parser.lower() in ('4column', '4col'):
    options.parser = torch.measure.load.split_four_column
  elif options.parser.lower() in ('5column', '5col'):
    options.parser = torch.measure.load.split_five_column
  else: #try an import
    if options.parser.find('.') == -1:
      parser.error("parser module should be either '4column', '5column' or a valid python function identifier in the format 'module.function': '%s' is invalid" % options.parser)

    mod, fct = options.parser.rsplit('.', 2)
    import imp
    try:
      fp, pathname, description = imp.find_module(mod, ['.'] + sys.path)
    except Exception, e:
      parser.error("import error for '%s': %s" % (options.parser, e))

    try:
      pmod = imp.load_module(mod, fp, pathname, description)
      options.parser = getattr(pmod, fct)
    except Exception, e:
      parser.error("loading error for '%s': %s" % (options.parser, e))
    finally:
      fp.close()

  if len(args) != 0:
    parser.error("this program does not accept positional arguments")

  return options

if __name__ == '__main__':
  options = get_options()

  neg, pos = options.parser(options.ifile)
  apthres(neg, pos, options.thres)
  sys.exit(0)
