#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 28 Jun 2011 12:54:06 CEST 

"""Contains a set of management utilities for a centralized driver script.
"""
  
import os
import sys

def load_db_module(name):
  """Loads a given database module, if it exists"""
  
  exec("from . import %s as module" % name)
  return module

def add_argument(name, subparsers):
  """Finds all catalogued commands for a certain DB"""

  try:
    module = load_db_module(name)
  except ImportError:
    return False

  if hasattr(module, 'add_commands'): 
    module.add_commands(subparsers)
    return True

  return False

def create_parser(**kwargs):
  """Creates a parser for the central manager taking into consideration the
  options for every module that can provide those."""

  import argparse

  parser = argparse.ArgumentParser(**kwargs)
  subparsers = parser.add_subparsers(title='databases')

  dirname = os.path.dirname(__file__)
  for k in os.listdir(dirname):
    d = os.path.join(dirname, k)
    if not os.path.isdir(d): continue
    if os.path.exists(os.path.join(dirname, k, '__init__.py')):
      add_argument(k, subparsers)

  return parser
