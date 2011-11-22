#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 27 Jun 09:12:40 2011 

"""This example shows how to use the Iris Flower (Fisher's) Dataset to create
3-class classifier based on Neural Networks (Multi-Layer Perceptrons - MLP).
"""

import os
import sys
import torch
import optparse
import tempfile #for package tests
import numpy

def choose_matplotlib_iteractive_backend():
  """Little logic to get interactive plotting right on OSX and Linux"""

  import platform
  import matplotlib
  if platform.system().lower() == 'darwin': #we are on OSX
    matplotlib.use('macosx')
  else:
    matplotlib.use('GTKAgg')

def generate_testdata(data, target):
  """Concatenates all data in a single 2D array. Examples are encoded row-wise,
  features, column-wise. The same for the targets.
  """
  destsize = 0
  for d in data: destsize += len(d)
  retval = numpy.zeros((destsize, 4), 'float64')
  t_retval = numpy.zeros((destsize, target[0].shape[0]), 'float64')
  retval.fill(0)
  cur = 0
  for k, d in enumerate(data):
    retval[cur:(cur+len(d)),:] = d.cat()
    for i in range(len(d)):
      t_retval[i+cur,:] = target[k]
    cur += len(d)
  return retval, t_retval

def create_machine(data, training_steps):
  """Creates the machine given the training data"""

  mlp = torch.machine.MLP((4, 4, len(data)))
  mlp.activation = torch.machine.Activation.TANH
  mlp.randomize() #reset weights and biases to a value between -0.1 and +0.1
  BATCH = 50
  trainer = torch.trainer.MLPRPropTrainer(mlp, BATCH)
  trainer.trainBiases = True #this is the default, but just to clarify!

  targets = [ #we choose the approximate Fisher response!
      numpy.array([+1., -1., -1.]), #setosa
      numpy.array([-1., +1., -1.]), #versicolor
      numpy.array([-1., -1., +1.]), #virginica
      ]

  # Associate the data to targets, by setting the arrayset order explicetly
  datalist = [data['setosa'], data['versicolor'], data['virginica']]

  # All data, as 2 x 2D arrays containing data and targets
  AllData, AllTargets = generate_testdata(datalist, targets)
  
  # A helper to select and shuffle the data
  S = torch.trainer.DataShuffler(datalist, targets)

  # We now iterate for several steps, look for the convergence
  retval = [torch.machine.MLP(mlp)]

  for k in range(training_steps):
    
    input, target = S(BATCH)

    # We use "train_" which is unchecked and faster. Use train() if you want
    # checks! See the MLPRPropTrainer documentation for details on this before
    # choosing the wrong approach.
    trainer.train_(mlp, input, target)
    print "|RMSE| @%d:" % k,
    print numpy.linalg.norm(torch.measure.rmse(mlp(AllData), AllTargets))
    retval.append(torch.machine.MLP(mlp))

  return retval #all machines => nice plotting!

def process_data(machine, data):
  """Iterates over classes and passes data through the trained machine"""
  
  output = {}
  for cl in data.keys():
    output[cl]=data[cl].foreach(machine.forward)

  return output

def vcat(a1, a2):
  """Concatenates 2 1D arrays"""
  retval = a1.__class__(a1.shape[0] + a2.shape[0])
  retval[:a1.shape[0]] = a1
  retval[a1.shape[0]:] = a2
  return retval

def plot(output):
  """Plots each of the outputs, with the classes separated by colors.
  """

  import matplotlib.pyplot as mpl

  histo = [{}, {}, {}]
  for k in output.keys():
    for i in range(len(histo)):
      histo[i][k] = output[k].cat()[i,:]

  order = ['setosa', 'versicolor', 'virginica']
  color = ['green', 'blue', 'red']

  FAR = []
  FRR = []
  THRES = []

  # Calculates separability
  for i, O in enumerate(order):
    positives = histo[i][O]
    negatives = vcat(*[histo[i][k] for k in order if k != O])
    # note: threshold a posteriori! (don't do this at home, kids ;-)
    thres = torch.measure.eerThreshold(negatives, positives)
    far, frr = torch.measure.farfrr(negatives, positives, thres)
    FAR.append(far)
    FRR.append(frr)
    THRES.append(thres)

  # Plots the class histograms
  plot_counter = 0
  for O, C in zip(order, color):
    for k in range(len(histo)):
      plot_counter += 1
      mpl.subplot(len(histo), len(order), plot_counter)
      mpl.hist(histo[k][O], bins=20, color=C, range=(-1,+1), label='Setosa', alpha=0.5)
      mpl.vlines((THRES[k],), 0, 60, colors=('red',), linestyles=('--',))
      mpl.axis([-1.1,+1.1,0,60])
      mpl.grid(True)
      if k == 0: mpl.ylabel("Data %s" % O.capitalize())
      if O == order[-1]: mpl.xlabel("Output %s" % order[k].capitalize())
      if O == order[0]: mpl.title("EER = %.1f%%" % (100*(FAR[k] + FRR[k])/2))

def fig2bzarray(fig):
  """
  @brief Convert a Matplotlib figure to a 3D blitz array with RGB channels and
  return it
  @param fig a matplotlib figure
  @return a blitz 3D array of RGB values
  """
  import numpy

  # draw the renderer
  fig.canvas.draw()

  # Get the RGB buffer from the figure, re-shape it adequately
  w,h = fig.canvas.get_width_height()
  buf = numpy.fromstring(fig.canvas.tostring_rgb(),dtype=numpy.uint8)
  buf.shape = (h,w,3)
  buf = numpy.transpose(buf, (2,0,1))

  return numpy.array(buf)

def makemovie(machines, data, filename=None):
  """Plots each of the outputs, with the classes separated by colors.
  """

  if not filename: 
    choose_matplotlib_iteractive_backend()
  else:
    import matplotlib
    matplotlib.use('Agg') #non-interactive avoids exception on display

  import matplotlib.pyplot as mpl

  output = None
  orows = 0
  ocols = 0
  if not filename: #start interactive plot animation
    mpl.ion()
  else:
    # test output size
    processed = process_data(machines[0], data)
    plot(processed)
    refimage = fig2bzarray(mpl.gcf())
    orows = 2*(refimage.shape[1]/2)
    ocols = 2*(refimage.shape[2]/2)
    output = torch.io.VideoWriter(filename, orows, ocols, 5) #5 Hz
    print "Saving %d frames to to %s" % (len(machines), filename)

  for i, k in enumerate(machines):
    # test output size
    processed = process_data(k, data)
    mpl.clf()
    plot(processed)
    mpl.suptitle("Fisher Iris DB / MLP Training step %d" % i)
    if not filename: mpl.draw() #updates ion drawing
    else:
      image = fig2bzarray(mpl.gcf())
      output.append(image[:,:orows,:ocols])
      sys.stdout.write('.')
      sys.stdout.flush()
    
  if filename:
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
  
  parser = optparse.OptionParser(description=__doc__,
      usage='%prog [--file=FILE]')
  parser.add_option("-t", "--steps", dest="steps", default=10, type=int,
      help="how many training times to train the MLP",
      metavar="INT")
  parser.add_option("-f", "--file", dest="filename", default=None,
      help="write plot movie to FILE (implies non-interactiveness)",
      metavar="FILE")
  parser.add_option("--self-test",
      action="store_true", dest="selftest", default=False,
      help=optparse.SUPPRESS_HELP)
  options, args = parser.parse_args()

  # Loads the dataset and performs LDA
  data = torch.db.iris.data()
  machines = create_machine(data, options.steps)

  if options.selftest:
    (fd, filename) = tempfile.mkstemp('.avi', 'torchtest_')
    os.close(fd)
    os.unlink(filename)
    makemovie(machines, data, filename)
    os.unlink(filename)
  else:
    makemovie(machines, data, options.filename)

  sys.exit(0)
