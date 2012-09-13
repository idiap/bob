#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.machine
"""

import unittest
import bob

class ExampleTest(unittest.TestCase):

  def test01_xml2hdf5(self):

    import os

    from pkg_resources import resource_filename
    net = resource_filename(__name__, os.path.join('data', 'network.xml'))

    from bob.machine.example.xml2hdf5 import main
    cmdline = ['--self-test', net]
    self.assertEqual(main(cmdline), 0)

  def test02_xml2hdf5(self):

    import os

    from pkg_resources import resource_filename
    net = resource_filename(__name__, os.path.join('data',
      'network-without-bias.xml'))

    from bob.machine.example.xml2hdf5 import main
    cmdline = ['--self-test', net]
    self.assertEqual(main(cmdline), 0)
