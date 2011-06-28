#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 27 Jun 09:12:40 2011 

"""This example shows how to use the Iris Flower (Fisher's) Dataset to create
3-class classifier based on Linear Discriminant Analysis.

Note: This example will consider all 3 classes for the LDA training. This is
*not* what Fisher did on his paper entitled "The Use of Multiple Measurements
in Taxonomic Problems", Annals of Eugenics, pp. 179-188, 1936. In that work
Fisher does the "right" thing only for the first 2-class problem (setosa x
versicolor). You can reproduce the 2-class LDA using Torch's LDA training
system w/o problems. When inserting the virginica class, Fisher decides for a
different metric (4vi + ve -5se) and solves lambda for the matrices in the last
row of Table VIII. 

This is OK, but does not generalize the method proposed on the begining of his
paper. Results achieved by the generalized LDA method will not match Fisher's
result on that last table, be aware. That being said, the final histogram
presented on that paper looks quite similar to the one produced by this script,
showing that Fisher's solution was approximately correct.
"""

import os
import sys
import torch
import optparse
import tempfile #for package tests

def choose_matplotlib_iteractive_backend():
  """Little logic to get interactive plotting right on OSX and Linux"""

  import platform
  import matplotlib
  if platform.system().lower() == 'darwin': #we are on OSX
    matplotlib.use('macosx')
  else:
    matplotlib.use('GTKAgg')

def create_machine(data):
  """Creates the machine given the training data"""

  lda = torch.trainer.FisherLDATrainer()
  machine, eigenValues = lda.train(data.values())

  return machine

def process_data(machine, data):
  """Iterates over classes and passes data through the trained machine"""
  
  output = {}
  for cl in data.keys():
    output[cl]=data[cl].foreach(machine.forward)

  return output

def plotting(output, filename=None):
  """Cherry picks the first variable (most discriminant) and reproduces the
     histogram plot Fisher has on this paper, last page.
  """

  if not filename: choose_matplotlib_iteractive_backend()
  else:
    import matplotlib
    matplotlib.use('pdf') #non-interactive avoids exception on display

  import matplotlib.pyplot as mpl

  histo = {}
  for k in output.keys(): histo[k] = output[k].cat()[0,:]

  # Plots the class histograms
  mpl.hist(histo['setosa'], bins=8, color='green', label='Setosa', alpha=0.5)
  mpl.hist(histo['versicolor'], bins=8, color='blue', label='Versicolor',
      alpha=0.5)
  mpl.hist(histo['virginica'], bins=8, color='red', label='Virginica', 
      alpha=0.5)
  mpl.legend()
  mpl.grid(True)
  mpl.axis([-3,+3,0,20])
  mpl.title("Iris Plants / 1st. LDA component")
  mpl.xlabel("LDA[0]")
  mpl.ylabel("Count")
  if filename: #running in a non-interactive way, save the file
    mpl.savefig(filename)
  else: #running in an interactive way, show the plot @ the user screen
    mpl.show()

if __name__ == '__main__':
  
  parser = optparse.OptionParser(description=__doc__,
      usage='%prog [--file=FILE]')
  parser.add_option("-f", "--file", dest="filename", default=None,
      help="write plots to FILE (implies non-interactiveness)",
      metavar="FILE")
  parser.add_option("--self-test",
      action="store_true", dest="selftest", default=False,
      help=optparse.SUPPRESS_HELP)
  options, args = parser.parse_args()

  # Loads the dataset and performs LDA
  data = torch.db.iris.data()
  machine = create_machine(data)
  output = process_data(machine, data)

  if options.selftest:
    (fd, filename) = tempfile.mkstemp('.pdf', 'torchtest_')
    os.close(fd)
    os.unlink(filename)
    plotting(output, filename)
    os.unlink(filename)
  else:
    plotting(output, options.filename)

  sys.exit(0)
