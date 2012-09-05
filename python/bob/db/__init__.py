#!/usr/bin/env python
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 23 Jun 20:22:28 2011 CEST 
# vim: set fileencoding=utf-8 :

"""The db package contains simplified APIs to access data for various databases
that can be used in Biometry, Machine Learning or Pattern Classification."""

from . import utils, driver, iris

__all__ = dir()
