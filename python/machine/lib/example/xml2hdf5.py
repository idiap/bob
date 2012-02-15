#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Jul 8 17:11:44 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

"""Converts an old NeuralLab XML file to the new HDF5 format. Use this as an
example if you have your own neural configuration you would like to get
converted to an HDF5-compatible neural network representation.

.. note::

  bob currently only supports fully-connected feed-forward networks.

.. note::

  This script only supports reading configurations with a single hidden layer.
"""

from xml.dom.minidom import parse as xml_parse
import bob
import numpy

def acttext_to_enum(s):
  """Converts the XML activation representation to a bob value"""
  
  if s.lower() == 'tanh':
    return bob.machine.Activation.TANH
  elif s.lower() == 'sigmoid':
    return bob.machine.Activation.LOG
  elif s.lower() == 'linear':
    return bob.machine.Activation.LINEAR
  else:
    raise RuntimeError, "unsupported activation %s" % s

def set_input(mach, inputs, verbose):
  """read input configuration, sets values on the machine"""

  input_subtract = []
  input_divide = []
  identities = []

  if verbose: print "INPUT: %d found" % len(inputs)
  for c in inputs:
    identity = int(c.getAttribute('id'))
    input_subtract.append(float(c.getAttribute('subtract')))
    input_divide.append(float(c.getAttribute('divide')))
    identities.append(identity)
    if verbose:
      print "  Input %d, sub: %.5e; div: %.5e" % (identity,
          input_subtract[-1], input_divide[-1])

  mach.input_subtract = numpy.array(input_subtract)
  mach.input_divide = numpy.array(input_divide)

  return identities

def set_hidden(mach, hidden, verbose):
  """read hidden layer neurons and sets values on the machine"""
 
  identities = []
  activation = None

  if verbose:
    print "HIDDEN: %d found" % len(hidden)
  for c in hidden:
    identity = int(c.getAttribute('id'))
    actfun = c.getElementsByTagName('backPropagation')[0].getAttribute('activationFunction')
    if activation is None:
      activation = acttext_to_enum(actfun)
      mach.activation = activation
      if verbose: print "  Activation set to %s" % mach.activation
    else:
      if activation != acttext_to_enum(actfun):
        raise RuntimeError, "mixed activation in hidden neuron %d" % identity
    identities.append(identity)
    if verbose:
      print "  Hidden %d, act: %s" % (identity, activation)

  return identities, activation

def set_output(mach, outputs, activation, verbose):
  """read output layer inforation and sets values on the machine"""

  identities = []

  # read output layer configuration
  if verbose:
    print "OUTPUT: %d found" % len(outputs)
  for c in outputs:
    identity = int(c.getAttribute('id'))
    actfun = c.getElementsByTagName('backPropagation')[0].getAttribute('activationFunction')
    if activation is None:
      activation = acttext_to_enum(actfun)
      mach.activation = activation
      if verbose: print "  Activation set to %s" % mach.activation
    else:
      if activation != acttext_to_enum(actfun):
        raise RuntimeError, "mixed activation in output neuron %d" % identity
    identities.append(identity)
    if verbose:
      print "  Output %d, act: %s" % (identity, activation)

  return identities, activation

def load_synapses(synapses, verbose):
  """Loads all interesting synapse information"""

  retval = []

  if verbose:
    print "SYNAPSES: %d found" % len(synapses)
  for c in synapses:
    identity = int(c.getAttribute('id'))
    fr = int(c.getAttribute('from'))
    to = int(c.getAttribute('to'))
    value = float(c.getElementsByTagName('weight')[0].childNodes[0].wholeText.strip())
    retval.append((fr, to, value))
    if verbose:
      print "  Synapse %d; %d => %d; value: %.5e" % (identity, fr, to, value)

  return retval

def organize_synapses(mach, inputs, hidden, outputs, bias, synapses):
  """Organizes the synapse values and set the machine weights and biases.
  
  Synapses received as input are organized as a tuple: (from, to, value)
  """

  # 1. re-organize the tuple input in a dictionary containing the ordered
  # values for each emitting node.
  emitter = {}
  for s in synapses:
    l = emitter.setdefault(s[0], {})
    l[s[1]] = s[2]

  # 2. emitting nodes are inputs, hidden or bias neurons; we start setting the
  # weight matrix from input -> hidden layer.
  weight0 = []
  for input_neuron in sorted(inputs):
    weight0.append([])
    for hidden_neuron in sorted(hidden):
      weight0[-1].append(emitter[input_neuron][hidden_neuron])
  weight0 = numpy.array(weight0)

  weight1 = []
  for hidden_neuron in sorted(hidden):
    weight1.append([])
    for output_neuron in sorted(outputs):
      weight1[-1].append(emitter[hidden_neuron][output_neuron])
  weight1 = numpy.array(weight1)

  mach.weights = [weight0, weight1]

  if bias is not None:
    hidden_bias = []
    for hidden_neuron in sorted(hidden):
      hidden_bias.append(emitter[bias][hidden_neuron])
    hidden_bias = numpy.array(hidden_bias)

    output_bias = []
    for output_neuron in sorted(outputs):
      output_bias.append(emitter[bias][output_neuron])
    output_bias = numpy.array(output_bias)

    mach.biases = [hidden_bias, output_bias]

def load_xml(filename, verbose):
  """Loads the XML and outputs a dictionary containing information about the
  network.
  """

  dom = xml_parse(filename)

  shape = []
  inputs = dom.getElementsByTagName('input')
  hidden = dom.getElementsByTagName('hidden')
  outputs = dom.getElementsByTagName('output')

  mach = bob.machine.MLP((len(inputs), len(hidden), len(outputs)))

  inputs = set_input(mach, inputs, verbose)
  hidden, activation = set_hidden(mach, hidden, verbose)
  outputs, activation = set_output(mach, outputs, activation, verbose)

  # read bias configuration
  bias = None
  c = dom.getElementsByTagName('bias')
  if c:
    bias = int(c[0].getAttribute('id'))
    if verbose: print "BIAS: present, neuron %d" % bias
  else:
    mach.biases = 0.0

  # load the synapses information
  synapses = load_synapses(dom.getElementsByTagName('synapse'), verbose)

  organize_synapses(mach, inputs, hidden, outputs, bias, synapses)

  return mach

def convert(ifile, ofile, verbose):
  """Runs the full conversion from XML to HDF5"""

  machine = load_xml(ifile, verbose)
  machine.save(bob.io.HDF5File(ofile))
  if verbose: 
    print "MLP/HDF5 machine saved at %s" % ofile
    print machine

def main():

  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('input', metavar='input', type=str,
      help='input network file to be converted to hdf5')
  parser.add_argument('output', metavar='output', type=str, nargs='?',
      help='output MLP filename (if not given defaults to the input filename with .hdf5 replacing .xml)')
  parser.add_argument('-v', '--verbose', dest='verbose', 
      default=False, action='store_true',
      help='makes the execution a bit more verbose')
  parser.add_argument('--self-test', dest='selftest', 
      default=False, action='store_true',
      help=argparse.SUPPRESS)

  args = parser.parse_args()

  if args.selftest:
    import os
    import sys
    import tempfile
    (fd, name) = tempfile.mkstemp(prefix="bob_example", suffix='.hdf5')
    os.close(fd)
    os.unlink(name)
    mydir = os.path.dirname(os.path.realpath(sys.argv[0]))
    convert(os.path.join(mydir, 'network.xml'), name, args.verbose)
    os.unlink(name)
    convert(os.path.join(mydir, 'network-without-bias.xml'), name, args.verbose)
    os.unlink(name)
    sys.exit(0)

  output = args.output
  if not output: output = os.path.splitext(args.input)[0] + '.hdf5'
  convert(args.input, output, args.verbose)
