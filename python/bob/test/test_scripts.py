#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 13 Sep 2012 15:47:19 CEST

"""Test scripts in bob
"""

import os
import unittest

class ScriptTest(unittest.TestCase):

  def test01_bob_config(self):
   
    from bob.script.config import main
    self.assertEqual(main(), 0)
