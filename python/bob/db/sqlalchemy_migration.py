#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 05 Jul 2011 12:59:34 CEST 

"""Implements the Enum type for sqlalchemy versions older than 0.6
"""

try:
  from sqlalchemy import Enum
except ImportError:
  from sqlalchemy import types

  class Enum(types.TypeDecorator):
      impl = types.Unicode
      
      def __init__(self, *values):
          """Emulates an Enum type.

          values:
             A list of valid values for this column
          """

          if values is None or len(values) is 0:
              raise AssertionError('Enum requires a list of values')
          self.values = values[:]

          # The length of the string/unicode column should be the longest string
          # in values
          size = max([len(v) for v in values if v is not None])
          super(Enum, self).__init__(size)        
          
          
      def process_bind_param(self, value, dialect):
          if value not in self.values:
              raise AssertionError('"%s" not in Enum.values' % value)
          return value
          
          
      def process_result_value(self, value, dialect):
          return value

try:
  from sqlalchemy.orm import relationship
except ImportError:
  from sqlalchemy.orm import relation as relationship

__all__ = ['Enum', 'relationship']
