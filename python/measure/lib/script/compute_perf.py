#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed May 25 13:27:46 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This script runs error analysis on the development and test set scores, in a
four column format: 
  1. Computes the threshold using either EER or min. HTER criteria on
     develoment set scores;
  2. Applies the above threshold on test set scores to compute the HTER
  3. Plots ROC, EPC and DET curves to a multi-page PDF file
"""

__epilog__ = """
Examples:

  1. Specify a different output filename

     $ %(prog)s --output=mycurves.pdf --devel=dev.scores --test=test.scores

  2. Specify a different number of points 

     $ %(prog)s --points=500 --devel=dev.scores --test=test.scores

  3. Don't plot (only calculate thresholds)

     $ %(prog)s --no-plot --devel=dev.scores --test=test.scores

Note: 

This is just an example program. It is not meant to be perfect or complete,
just to give you the basis to develop your own scripts to plot things in the
way you need. In order to tweak more options, just copy this file to your
directory and modify it to fit your needs. You can easily copy this script like
this:

  $ cp `which %(prog)s` .
  $ vim %(prog)s
"""

import sys, os, bob

def print_crit(dev_neg, dev_pos, test_neg, test_pos, crit):
  """Prints a single output line that contains all info for a given criterium"""

  if crit == 'EER':
    thres = bob.measure.eerThreshold(dev_neg, dev_pos)
  else:
    thres = bob.measure.minHterThreshold(dev_neg, dev_pos)

  dev_far, dev_frr = bob.measure.farfrr(dev_neg, dev_pos, thres)
  dev_hter = (dev_far + dev_frr)/2.0

  test_far, test_frr = bob.measure.farfrr(test_neg, test_pos, thres)
  test_hter = (test_far + test_frr)/2.0

  print("[Min. criterium: %s] Threshold on Development set: %e" % (crit, thres))
  
  dev_ni = dev_neg.shape[0] #number of impostors
  dev_fa = int(round(dev_far*dev_ni)) #number of false accepts
  dev_nc = dev_pos.shape[0] #number of clients
  dev_fr = int(round(dev_frr*dev_nc)) #number of false rejects
  test_ni = test_neg.shape[0] #number of impostors
  test_fa = int(round(test_far*test_ni)) #number of false accepts
  test_nc = test_pos.shape[0] #number of clients
  test_fr = int(round(test_frr*test_nc)) #number of false rejects

  dev_far_str = "%.3f%% (%d/%d)" % (100*dev_far, dev_fa, dev_ni)
  test_far_str = "%.3f%% (%d/%d)" % (100*test_far, test_fa, test_ni)
  dev_frr_str = "%.3f%% (%d/%d)" % (100*dev_frr, dev_fr, dev_nc)
  test_frr_str = "%.3f%% (%d/%d)" % (100*test_frr, test_fr, test_nc)
  dev_max_len = max(len(dev_far_str), len(dev_frr_str))
  test_max_len = max(len(test_far_str), len(test_frr_str))

  def fmt(s, space):
    return ('%' + ('%d' % space) + 's') % s

  print("       | %s | %s" % (fmt("Development", -1*dev_max_len), 
    fmt("Test", -1*test_max_len)))
  print("-------+-%s-+-%s" % (dev_max_len*"-", (2+test_max_len)*"-"))
  print("  FAR  | %s | %s" % (fmt(dev_far_str, dev_max_len), fmt(test_far_str,
    test_max_len)))
  print("  FRR  | %s | %s" % (fmt(dev_frr_str, dev_max_len), fmt(test_frr_str,
    test_max_len)))
  dev_hter_str = "%.3f%%" % (100*dev_hter)
  test_hter_str = "%.3f%%" % (100*test_hter)
  print("  HTER | %s | %s" % (fmt(dev_hter_str, -1*dev_max_len), 
    fmt(test_hter_str, -1*test_max_len)))

def plots(dev_neg, dev_pos, test_neg, test_pos, npoints, filename):
  """Saves ROC, DET and EPC curves on the file pointed out by filename."""

  import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
  import matplotlib.pyplot as mpl
  from matplotlib.backends.backend_pdf import PdfPages

  pp = PdfPages(filename)

  # ROC
  fig = mpl.figure()
  bob.measure.plot.roc(dev_neg, dev_pos, npoints, color=(0.3,0.3,0.3), 
      linestyle='--', dashes=(6,2), label='development')
  bob.measure.plot.roc(test_neg, test_pos, npoints, color=(0,0,0),
      linestyle='-', label='test')
  mpl.axis([0,40,0,40])
  mpl.title("ROC Curve")
  mpl.xlabel('FRR (%)')
  mpl.ylabel('FAR (%)')
  mpl.grid(True, color=(0.3,0.3,0.3))
  mpl.legend()
  pp.savefig(fig)

  # DET
  fig = mpl.figure()
  bob.measure.plot.det(dev_neg, dev_pos, npoints, color=(0.3,0.3,0.3), 
      linestyle='--', dashes=(6,2), label='development')
  bob.measure.plot.det(test_neg, test_pos, npoints, color=(0,0,0),
      linestyle='-', label='test')
  bob.measure.plot.det_axis([0.01, 40, 0.01, 40])
  mpl.title("DET Curve")
  mpl.xlabel('FRR (%)')
  mpl.ylabel('FAR (%)')
  mpl.grid(True, color=(0.3,0.3,0.3))
  mpl.legend()
  pp.savefig(fig)

  # EPC
  fig = mpl.figure()
  bob.measure.plot.epc(dev_neg, dev_pos, test_neg, test_pos, npoints, 
      color=(0,0,0), linestyle='-')
  mpl.title('EPC Curve')
  mpl.xlabel('Cost')
  mpl.ylabel('Min. HTER (%)')
  mpl.grid(True, color=(0.3,0.3,0.3))
  pp.savefig(fig)

  pp.close()

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

  parser.add_option('-d', '--devel', dest="dev", default=None,
      help="Name of the file containing the development scores (defaults to %default)", metavar="FILE")
  parser.add_option('-t', '--test', dest="test", default=None,
      help="Name of the file containing the test scores (defaults to %default)", metavar="FILE")
  parser.add_option('-n', '--points', dest="npoints", default=100, type="int",
      help="Number of points to use in the curves (defaults to %default)",
      metavar="INT(>0)")
  parser.add_option('-o', '--output', dest="plotfile", default="curves.pdf",
      help="Name of the output file that will contain the plots (defaults to %default)", metavar="FILE")
  parser.add_option('-x', '--no-plot', dest="doplot", default=True,
      action='store_false', help="If set, then I'll execute no plotting")
  parser.add_option('-p', '--parser', dest="parser", default="4column",
      help="Name of a known parser or of a python-importable function that can parse your input files and return a tuple (negatives, positives) as blitz 1-D arrays of 64-bit floats. Consult the API of bob.measure.load.split_four_column() for details", metavar="NAME.FUNCTION")
  
  # This option is not normally shown to the user...
  parser.add_option("--self-test",
      action="store_true", dest="selftest", default=False,
      help=optparse.SUPPRESS_HELP)
      #help="if set, runs an internal verification test and erases any output")

  options, args = parser.parse_args()

  if options.selftest:
    # then we go into test mode, all input is preset
    import tempfile
    outputdir = tempfile.mkdtemp()
    options.plotfile = os.path.join(outputdir, "curves.pdf")

  if options.dev is None:
    parser.error("you should give a development score set with --devel")

  if options.test is None:
    parser.error("you should give a test score set with --test")

  #parse the score-parser
  if options.parser.lower() in ('4column', '4col'):
    options.parser = bob.measure.load.split_four_column
  elif options.parser.lower() in ('5column', '5col'):
    options.parser = bob.measure.load.split_five_column
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

def main():
  options = get_options()

  dev_neg, dev_pos = options.parser(options.dev)
  test_neg, test_pos = options.parser(options.test)

  print_crit(dev_neg, dev_pos, test_neg, test_pos, 'EER')
  print_crit(dev_neg, dev_pos, test_neg, test_pos, 'Min. HTER')
  if options.doplot:
    plots(dev_neg, dev_pos, test_neg, test_pos, 
        options.npoints, options.plotfile)
    print("[Plots] Performance curves => '%s'" % options.plotfile)

  if options.selftest: #remove output file + tmp directory
    import shutil
    shutil.rmtree(os.path.dirname(options.plotfile))

  sys.exit(0)
