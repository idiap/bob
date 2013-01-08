#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 23 May 2011 16:23:05 CEST

"""A set of utilities to load score files with different formats.
"""

import numpy

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

  # read four column list
  scores_list = four_column(filename)

  # split in positives and negatives
  neg = []
  pos = []
  for (client_id, probe_id, _, score_str) in scores_list:
    try:
      score = float(score_str)
      if client_id == probe_id:
        pos.append(score)
      else:
        neg.append(score)
    except:
      raise SyntaxError, 'Cannot convert score "%s" to float' % score_str

  return (numpy.array(neg, numpy.float64), numpy.array(pos, numpy.float64))

def cmc_four_column(filename):
  """Loads scores to compute CMC curves from a file in four column format.
  The four column file needs to be in the same format as described in the four_column function,
  and the "test label" (column 3) has to contain the test/probe file name.

  This function returns a list of tuples.
  For each probe file, the tuple consists of a list of negative scores and a list of positive scores.
  Usually, the list of positive scores should contain only one element, but more are allowed.

  The result of this function can directly be passed to, e.g., the bob.measure.cmc function.
  """
  # read four column list
  all_list = four_column(filename)
  # extract positives and negatives
  pos_dict = {}
  neg_dict = {}
  for (client_id, probe_id, probe_name, score_str) in all_list:
    try:
      score = float(score_str)
      # check in which dict we have to put the score
      if client_id == probe_id:
        correct_dict = pos_dict
      else:
        correct_dict = neg_dict
      # append score
      if probe_name in correct_dict:
        correct_dict[probe_name].append(score)
      else:
        correct_dict[probe_name] = [score]
    except:
      raise SyntaxError, "Cannot convert score '%s' to float" % score_str

  # convert to lists of tuples of ndarrays
  retval = []
  import logging
  logger = logging.getLogger('bob')
  for probe_name in sorted(pos_dict.keys()):
    if probe_name in neg_dict:
      retval.append((numpy.array(neg_dict[probe_name], numpy.float64), numpy.array(pos_dict[probe_name], numpy.float64)))
    else:
      logger.warn('For probe name "%s" there are only positive scores. This probe name is ignored.' % probe_name)
  # test if there are probes for which only negatives exist
  for probe_name in sorted(neg_dict.keys()):
    if not probe_name in pos_dict.keys():
       logger.warn('For probe name "%s" there are only negative scores. This probe name is ignored.' % probe_name)

  return retval

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

  # read five column list
  scores_list = five_column(filename)

  # split in positives and negatives
  neg = []
  pos = []
  for (client_id, _, probe_id, _, score_str) in scores_list:
    try:
      score = float(score_str)
      if client_id == probe_id:
        pos.append(score)
      else:
        neg.append(score)
    except:
      raise SyntaxError, 'Cannot convert score "%s" to float' % score_str

  return (numpy.array(neg, numpy.float64), numpy.array(pos, numpy.float64))

def cmc_five_column(filename):
  """Loads scores to compute CMC curves from a file in five column format.
  The four column file needs to be in the same format as described in the five_column function,
  and the "test label" (column 4) has to contain the test/probe file name.

  This function returns a list of tuples.
  For each probe file, the tuple consists of a list of negative scores and a list of positive scores.
  Usually, the list of positive scores should contain only one element, but more are allowed.

  The result of this function can directly be passed to, e.g., the bob.measure.cmc function.
  """
  # read four column list
  all_list = five_column(filename)

  pos_dict = {}
  neg_dict = {}
  for (client_id, _, probe_id, probe_name, score_str) in all_list:
    try:
      score = float(score_str)
      # check in which dict we have to put the score
      if client_id == probe_id:
        correct_dict = pos_dict
      else:
        correct_dict = neg_dict
      # append score
      if probe_name in correct_dict:
        correct_dict[probe_name].append(score)
      else:
        correct_dict[probe_name] = [score]
    except:
      raise SyntaxError, 'Cannot convert score "%s" to float' % score_str

  # convert to lists of tuples of ndarrays
  retval = []
  import logging
  logger = logging.getLogger('bob')
  for probe_name in sorted(pos_dict.keys()):
    if probe_name in neg_dict:
      retval.append((numpy.array(neg_dict[probe_name], numpy.float64), numpy.array(pos_dict[probe_name], numpy.float64)))
    else:
      logger.warn('For probe name "%s" there are only positive scores. This probe name is ignored.' % probe_name)
  # test if there are probes for which only negatives exist
  for probe_name in sorted(neg_dict.keys()):
    if not probe_name in pos_dict.keys():
       logger.warn('For probe name "%s" there are only negative scores. This probe name is ignored.' % probe_name)
  return retval
