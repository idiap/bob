#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 23 May 2011 16:23:05 CEST 

"""A set of utilities to load score files with different formats.
"""

def four_column(filename):
  """Loads a score set from a single file to memory. 

  Verifies that all fields are correctly placed and contain valid fields.

  Returns a python list of tuples containg the following fields:

    [0]
      claimed identity (string)
    [1]
      real identity (string)
    [2]
      test label (string)
    [3]
      score (float)
  """

  retval = []
  for i, l in enumerate(open(filename, 'rt')):
    s = l.strip()
    if len(s) == 0 or s[0] == '#': continue #empty or comment
    field = [k.strip() for k in s.split()]
    if len(field) < 4: 
      raise SyntaxError, 'Line %d of file "%s" is invalid: %s' % \
          (i, filename, l)
    try:
      score = float(field[3])
      t = (field[0], field[1], field[2], score)
      retval.append(t)
    except:
      raise SyntaxError, 'Cannot convert score to float at line %d of file "%s": %s' % (i, filename, l)
 
  return retval

def split_four_column(filename):
  """Loads a score set from a single file to memory and splits the scores
  between positives and negatives. The score file has to respect the 4 column
  format as defined in the method four_column().

  This method avoids loading and allocating memory for the strings present in
  the file. We only keep the scores.

  Returns a python tuple (negatives, positives). The values are 1-D blitz
  arrays of float64.
  """
  
  from ..core.array import array
  
  neg = []
  pos = []
  for i, l in enumerate(open(filename, 'rt')):
    s = l.strip()
    if len(s) == 0 or s[0] == '#': continue #empty or comment
    field = [k.strip() for k in s.split()]
    if len(field) < 4: 
      raise SyntaxError, 'Line %d of file "%s" is invalid: %s' % \
          (i, filename, l)
    try:
      score = float(field[3])
      if field[0] == field[1]: pos.append(score)
      else: neg.append(score)
    except:
      raise SyntaxError, 'Cannot convert score to float at line %d of file "%s": %s' % (i, filename, l)

  return (array(neg, 'float64'), array(pos, 'float64'))

def five_column(filename):
  """Loads a score set from a single file to memory. 

  Verifies that all fields are correctly placed and contain valid fields.

  Returns a python list of tuples containg the following fields:

    [0]
      claimed identity (string)
    [1]
      model label (string)
    [2]
      real identity (string)
    [3]
      test label (string)
    [4]
      score (float)
  """

  retval = []
  for i, l in enumerate(open(filename, 'rt')):
    s = l.strip()
    if len(s) == 0 or s[0] == '#': continue #empty or comment
    field = [k.strip() for k in s.split()]
    if len(field) < 5: 
      raise SyntaxError, 'Line %d of file "%s" is invalid: %s' % \
          (i, filename, l)
    try:
      score = float(field[4])
      t = (field[0], field[1], field[2], field[3], score)
      retval.append(t)
    except:
      raise SyntaxError, 'Cannot convert score to float at line %d of file "%s": %s' % (i, filename, l)
 
  return retval

def split_five_column(filename):
  """Loads a score set from a single file to memory and splits the scores
  between positives and negatives. The score file has to respect the 5 column
  format as defined in the method five_column().

  This method avoids loading and allocating memory for the strings present in
  the file. We only keep the scores.

  Returns a python tuple (negatives, positives). The values are 1-D blitz
  arrays of float64.
  """

  from ..core.array import array

  neg = []
  pos = []
  for i, l in enumerate(open(filename, 'rt')):
    s = l.strip()
    if len(s) == 0 or s[0] == '#': continue #empty or comment
    field = [k.strip() for k in s.split()]
    if len(field) < 5: 
      raise SyntaxError, 'Line %d of file "%s" is invalid: %s' % \
          (i, filename, l)
    try:
      score = float(field[4])
      if field[0] == field[2]: pos.append(score)
      else: neg.append(score)
    except:
      raise SyntaxError, 'Cannot convert score to float at line %d of file "%s": %s' % (i, filename, l)
 
  return (array(neg, 'float64'), array(pos, 'float64'))

