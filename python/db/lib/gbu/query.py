#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Fri May 11 17:20:46 CEST 2012
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

"""This module provides the Database interface allowing the user to query the
GBU database in the most obvious ways.
"""

from .. import utils
from .models import *
from . import dbname

class Database(object):
  """The dataset class opens and maintains a connection opened to the Database.

  It provides many different ways to probe for the characteristics of the data
  and for the data itself inside the database.
  """

  def __init__(self):
    # opens a session to the database - keep it open until the end
    self.m_session = utils.session(dbname())
    self.m_groups  = ('world', 'dev') # GBU does not provide an eval set
    self.m_train_sets = Trainset.m_names # Will be queried by the 'subworld' parameters
    self.m_purposes = Protocol.m_purposes
    self.m_protocols = Protocol.m_names
    self.m_types = Protocol.m_types # The type of protocols: The default GBU or one with multiple files per model 

  def __check_validity__(self, elements, description, possibilities, default = None):
    """Checks validity of user input data against a set of valid values"""
    if not elements: 
      return default if default else possibilities 
    if not isinstance(elements, list) and not isinstance(elements, tuple): 
      return self.__check_validity__((elements,), description, possibilities, default)
    for k in elements:
      if k not in possibilities:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (description, k, possibilities)
    return elements
  
  def __check_single__(self, element, description, possibilities, default = None):
    """Checks validity of user input data against a set of valid values"""
    if not element:
      return default
    if isinstance(element,tuple) or isinstance (element,list):
      if len(element) > 1:
        raise RuntimeError, 'For %s, only single elements from %s are allowed' % (description, possibilities)
      element = element[0]
    if element not in possibilities:
      raise RuntimeError, 'The given %s "%s" is not allowed. Please choose one of %s' % (description, element, possibilities)
    return element

  def __make_path__(self, file, directory, extension):
    """Generates the file name for the given File object."""
    stem = os.path.join(file.m_directory, file.m_filename)
    if not extension: extension = ''
    if directory: return os.path.join(directory, stem + extension)
    return stem + extension
    

  def clients(self, groups=None, subworld=None, protocol=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    groups
     One or several groups to which the models belong ('world', 'dev').
    
    subworld
      One or several training sets ('x1', 'x2', 'x4', 'x8'), only valid if group is 'world'.
      
    protocol
      One or several of the GBU protocols ('Good', 'Bad', 'Ugly'), only valid if group is 'dev'.


    Returns: A list containing all the client id's which have the given
    properties.
    """

    groups = self.__check_validity__(groups, "group", self.m_groups)
    subworld = self.__check_validity__(subworld, "training set", self.m_train_sets)
    protocol = self.__check_validity__(protocol, "protocol", self.m_protocols)

    retval = []
    # List of the clients
    if 'world' in groups:
      q = self.m_session.query(Client).join(File).join(Trainset)\
              .filter(Trainset.m_name.in_(subworld))
      for client_id in [k.m_signature for k in q]: 
        retval.append(client_id)

    if 'dev' in groups:
      q = self.m_session.query(Client).join(File).join(Protocol)\
              .filter(Protocol.m_name.in_(protocol))\
              .filter(Protocol.m_purpose == 'enrol')
      for client_id in [k.m_signature for k in q]: 
        retval.append(client_id)

    return retval


  def models(self, type='multi', groups=None, subworld=None, protocol=None):
    """Returns a set of models for the specific query by the user.
    The returned list depends on the type:
    
    * for type 'gbu': A list containing file id's (there is one model per file)
    * for type 'multi': A list containing client id's (there is one model per client)  

    Keyword Parameters:
    
    type
      One protocol type from ('gbu', 'multi')

    groups
      One or several groups to which the models belong ('world', 'dev').
    
    subworld
      One or several training sets ('x1', 'x2', 'x4', 'x8'), only valid if group is 'world'.
      
    protocol
      One or several of the GBU protocols ('Good', 'Bad', 'Ugly'), only valid if group is 'dev'.
    
    Returns: A list containing all the model id's belonging to the given group.
    """

    type = self.__check_single__(type, "types", self.m_types, 'multi')
   
    if type == 'multi':
      return self.clients(groups, subworld, protocol)

    groups = self.__check_validity__(groups, "group", self.m_groups)
    subworld = self.__check_validity__(subworld, "training set", self.m_train_sets)
    protocol = self.__check_validity__(protocol, "protocol", self.m_protocols)

    retval = []
    # List of the 
    if 'world' in groups:
      q = self.m_session.query(File).join(Trainset)\
              .filter(Trainset.m_name.in_(subworld))
      for file_id in [k.m_presentation for k in q]: 
        retval.append(file_id)

    if 'dev' in groups:
      q = self.m_session.query(File).join(Protocol)\
              .filter(Protocol.m_name.in_(protocol))\
              .filter(Protocol.m_purpose == 'enrol')
      for file_id in [k.m_presentation for k in q]: 
        retval.append(file_id)

    return retval


  def get_client_id_from_model_id(self, model_id, type='multi'):
    """Returns the client_id attached to the given model_id.
    Dependent on the type, it is expected that
    
    * model_id is a file_id, when type is 'gbu'
    * model_id is a client_id, when type is 'multi'
    
    Keyword Parameters:

    model_id
      The model_id to consider
      
    type
      One protocol type from ('gbu', 'multi')
      

    Returns: The client_id attached to the given model_id
    """
   
    type = self.__check_single__(type, "types", self.m_types, 'multi')
    
    if type == 'multi':
      return model_id
    
    q = self.m_session.query(Client).join(File)\
            .filter(File.m_presentation == model_id)
            
    assert q.count() == 1
    return q.first().m_signature
    

  def get_client_id_from_file_id(self, file_id):
    """Returns the client_id (real client id) attached to the given file_id
    
    Keyword Parameters:

    file_id
      The file_id to consider

    Returns: The client_id attached to the given file_id
    """
    q = self.m_session.query(File)\
            .filter(File.m_presentation == file_id)
            
    assert q.count() == 1
    return q.first().m_signature


  def files(self, directory=None, extension=None, groups=None, subworld=None, protocol=None, purposes=None, model_ids=None, type='multi'):
    """Returns a dictionary from file_ids to file paths using the specified restrictions:

    Keyword Parameters:

    directory
      A directory name that will be prepended to all file paths

    extension
      A filename extension that will be appended to all file paths

    groups
      One or several groups to which the models belong ('world', 'dev').
    
    subworld
      One or several training sets ('x1', 'x2', 'x4', 'x8'), only valid if group is 'world'.
      
    protocol
      One or several of the GBU protocols ('Good', 'Bad', 'Ugly'), only valid if group is 'dev'.
    
    purposes
      One or several groups for which files should be retrieved ('enrol', 'probe'),
      only valid when the group is 'dev'·

    model_ids
      If given (as a list of model id's or a single one), only the files
      belonging to the specified model id is returned. The content of the model id
      is dependent on the type:
      
      * model_id is a file_id, when type is 'gbu'
      * model_id is a client_id, when type is 'multi'
      
    type
      One protocol type from ('gbu', 'multi'), only required when model_ids are specified

    """
    
    def filter_model(query, type, model_ids):
      if model_ids and len(model_ids):
        if type == 'gbu':
          # for GBU protocol type, model id's are file id's
          query = query.filter(File.m_presentation.in_(model_ids))
        else:
          # for multi protocol type, model id's are client id's
          query = query.filter(File.m_signature.in_(model_ids))
      return query
          

    # check that every parameter is as expected
    groups = self.__check_validity__(groups, "group", self.m_groups)
    subworld = self.__check_validity__(subworld, "subworld", self.m_train_sets)
    protocol = self.__check_validity__(protocol, "protocol", self.m_protocols)
    purposes = self.__check_validity__(purposes, "purpose", self.m_purposes)

    # This test would be nice, but it takes to much time...
#    model_ids = self.__check_validity__(model_ids, 'model id', self.models(type,groups,subworld,protocol),[])
    # so we do not check that the given model id's are valid... 
    if isinstance(model_ids, str):
      model_ids = (model_ids)
    
    type = self.__check_single__(type, 'protocol type', self.m_types)

    retval = {}

    if 'world' in groups:
      query = self.m_session.query(File).join(Trainset)\
                  .filter(Trainset.m_name.in_(subworld))
      query = filter_model(query, type, model_ids)

      for file in query:
        retval[file.m_presentation] = self.__make_path__(file, directory, extension)
    
    if 'dev' in groups:
      query = self.m_session.query(File).join(Protocol)\
                  .filter(Protocol.m_name.in_(protocol))\
                  .filter(Protocol.m_purpose.in_(purposes))
      query = filter_model(query, type, model_ids)

      for file in query:
        retval[file.m_presentation] = self.__make_path__(file, directory, extension)

    return retval


  def save_one(self, file_id, obj, directory, extension):
    """Saves a single object supporting the bob save() protocol.

    This method will call save() on the given object using the correct
    database filename stem for the given id.
    
    Keyword Parameters:

    file_id
      The file_id of the object.

    obj
      The object that needs to be saved, respecting the bob save() protocol.

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs().

    extension
      The extension determines the way the object will be saved. 
    """

    import os
    from ...io import save

    query = self.m_session.query(File).filter(File.m_presentation == file_id)
    assert query.count() == 1
    filename = self.__make_path__(query.first(), directory, extension) 
    utils.makedirs_safe(os.path.dirname(filename))
    save(obj, filename)

  def save(self, data, directory, extension):
    """This method takes a dictionary of data and 
    saves it respecting to the given directory.

    Keyword Parameters:

    data
      A dictionary from file_id to the actual data to be saved. 

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs()

    extension
      The extension determines the way the data will be saved.
    """    

    for key, value in data:
      self.save_one(key, value, directory, extension)
