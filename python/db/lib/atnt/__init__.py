#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Fri Apr 20 12:04:44 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

"""
The AT&T database is a small database to test algorithms on. It is freely downloadable from:
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
"""
import os
import sys
import numpy
from .. import utils
from .commands import add_commands

# Use this variable to tell bob_dbmanage.py all driver that there is nothing to
# download for this database.
__builtin__ = True

class Database(object):
  """Wrapper class for the AT&T database (http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).
  This class defines a simple protocol for training, enrolment and probe by splitting the few images of the database in a reasonable manner.""" 

  def __init__(self):
    """Initializes the AT&T database and defines a simple protocol for the AT&T database."""
    self.m_groups = ('train', 'test')
    self.m_purposes = ('enrol', 'probe')
    self.m_client_ids = set(range(1, 41))
    self.m_files = set(range(1, 11))
    self.m_training_clients = set([1,2,5,6,10,11,12,14,16,17,20,21,24,26,27,29,33,34,36,39])
    self.m_enrol_files = set([2,4,5,7,9])

  def dbname(self):
    """Calculates my own name automatically."""
    return os.path.basename(os.path.dirname(__file__))


  def __check_validity__(self, l, obj, valid, default):
    """Checks validity of user input data against a set of valid values."""
    if not l: return default
    elif isinstance(l, str) or isinstance(l, int): return self.__check_validity__([l], obj, valid, default) 
    for k in l:
      if k not in valid:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
    return l

  def __make_path__(self, client_id, file_id, directory, extension):
    """Generates the file name for the given client id and file id of the AT&T database."""
    stem = os.path.join("s" + str(client_id), str(file_id))
    if not extension: extension = ''
    if directory: return os.path.join(directory, stem + extension)
    return stem + extension


  def client_ids(self, groups = None):
    """Returns the vector of ids of the clients used in a given purpose
    
    Keyword Parameters:

    groups
      One of the groups 'train', 'test' or a tuple with both of them (which is the default).
    """

    VALID_GROUPS = self.m_groups
    groups = self.__check_validity__(groups, "group", VALID_GROUPS, VALID_GROUPS)
    
    ids = set()
    if 'train' in groups:
      ids |= self.m_training_clients
    if 'test' in groups:
      ids |= self.m_client_ids - self.m_training_clients
      
    return list(sorted(ids))
  
  
  def get_client_id_from_file_id(self, file_id):
    """Returns the client id from the given image id"""
    return (file_id-1) / len(self.m_files) + 1
    

  def files(self, directory=None, extension=None, client_ids=[], groups=None, purposes=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    client_ids
      The ids of the clients whose files need to be retrieved. Should be a list of integral numbers from [1,40]

    groups
      One of the groups 'train' or 'test' or a list with both of them (which is the default).

    purposes
      One of the purposes 'enrol' or 'probe' or a list with both of them (which is the default).
      This field is ignored when the group 'train' is selected.

    """

    # check if groups set are valid
    VALID_GROUPS = self.m_groups
    groups = self.__check_validity__(groups, "group", VALID_GROUPS, VALID_GROUPS)

    # collect the ids to retrieve
    ids = set(self.client_ids(groups))

    # check the desired client ids for sanity
    VALID_IDS = self.m_client_ids
    client_ids = set(self.__check_validity__(client_ids, "client_id", VALID_IDS, VALID_IDS))

    # calculate the intersection between the ids and the desired client ids
    ids = ids & client_ids

    # check that the groups are valid
    VALID_PURPOSES = self.m_purposes
    if 'test' in groups:
      purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES, VALID_PURPOSES)
    else:
      purposes = VALID_PURPOSES
      
    file_ids = set()
    if 'enrol' in purposes:
      file_ids |= self.m_enrol_files
    if 'probe' in purposes:
      file_ids |= self.m_files - self.m_enrol_files
    
    # go through the dataset and collect all desired files
    files = {}
    for client_id in ids:
      for file_id in file_ids:
        files[(client_id-1) * len(self.m_files) + file_id] = self.__make_path__(client_id, file_id, directory, extension)
              
    return files


    
