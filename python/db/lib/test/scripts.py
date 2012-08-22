#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various scripts for bob.db
"""

import unittest
import bob

class IrisDBScriptTest(unittest.TestCase):

  def test01_iris_dump(self):
   
    from bob.db.script.dbmanage import main
    cmdline = 'iris dump --self-test'
    self.assertEqual(main(cmdline.split()), 0)

  def test02_iris_dump(self):
    
    from bob.db.script.dbmanage import main
    cmdline = 'iris dump --class=versicolor --self-test'
    self.assertEqual(main(cmdline.split()), 0)

  def test03_iris_location(self):

    from bob.db.script.dbmanage import main
    self.assertEqual(main('iris location'.split()), 0)

  def test04_iris_version(self):

    from bob.db.script.dbmanage import main
    self.assertEqual(main('iris version'.split()), 0)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(IrisDBScriptTest)
