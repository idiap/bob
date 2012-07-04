#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Wed Jul  4 14:12:51 CEST 2012
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
AR face database.
"""

from .. import utils
from .models import *
from . import dbname

class Database(object):
  """The database class opens and maintains a connection opened to the Database.

  It provides many different ways to probe for the characteristics of the data
  and for the data itself inside the database.
  """

  def __init__(self):
    # opens a session to the database - keep it open until the end
    self.m_session = utils.session(dbname())
    # defines valid entries for various parameters
    self.m_groups  = Client.s_groups
    self.m_purposes = File.s_purposes
    self.m_genders = Client.s_genders
    self.m_sessions = File.s_sessions
    self.m_expressions = File.s_expressions
    self.m_illuminations = File.s_illuminations
    self.m_occlusions = File.s_occlusions
    self.m_protocols = Protocol.s_protocols

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

  def __make_path__(self, stem, directory, extension):
    """Generates the file name for the given File object."""
    if not extension: extension = ''
    if directory: return os.path.join(directory, stem + extension)
    return stem + extension
    

  def clients(self, groups=None, genders=None, protocol=None):
    """Returns a set of client ids for the specific query by the user.

    Keyword Parameters:

    groups
      One or several groups to which the models belong ('world', 'dev', 'eval').
      If not specified, all groups are returned.
    
    genders
      One of the genders ('m', 'w') of the clients.
      If not specified, clients of all genders are returned.
      
    protocol
      Ignored since clients are identical for all protocols. 

    Returns: A list containing all the client id's which have the given
    properties.
    """

    groups = self.__check_validity__(groups, "group", self.m_groups)
    genders = self.__check_validity__(genders, "group", self.m_genders)

    query = self.m_session.query(Client)\
                .filter(Client.m_group.in_(groups))\
                .filter(Client.m_gender.in_(genders))

    return [k.m_id for k in query]    

  # models() and clients() functions are identical 
  models = clients


  def get_client_id_from_file_id(self, file_id):
    """Returns the client_id (real client id) attached to the given file_id
    
    Keyword Parameters:

    file_id
      The file_id to consider

    Returns: The client_id attached to the given file_id
    """
    q = self.m_session.query(File)\
            .filter(File.m_id == file_id)
            
    assert q.count() == 1
    return q.first().m_client_id


  def objects(self, directory=None, extension=None, groups=None, protocol='all', purposes=None, model_ids=None, sessions=None, expressions=None, illuminations=None, occlusions=None, genders=None):
    """Using the specified restrictions, this function returns a dictionary from file_ids to a tuple containing:

    - 0: the resolved filename
    - 1: the model id (if one (and only one) model_id is given, it is copied here, otherwise the model id of the file)
    - 2: the claimed client id attached to the model (in case of the AR database: identical to 1)
    - 3: the real client id (for a probe image, the client id of the probe, otherwise identical to 1)
    - 4: the "stem" path (basename of the file; in case of the AR database: identical to the file id)

    Keyword Parameters:

    directory
      A directory name that will be prepended to all file paths

    extension
      A filename extension that will be appended to all file paths

    groups
      One or several groups to which the models belong ('world', 'dev', 'eval').
    
    protocol
      One of the AR protocols ('all', 'expression', 'illumination', 'occlusion', 'occlusion_and_illumination').
      This field is ignored for group 'world'.
    
    purposes
      One or several purposes for which files should be retrieved ('enrol', 'probe').
      This field is ignored for group 'world'.

    model_ids
      If given (as a list of model id's or a single one), only the files
      belonging to the specified model id is returned. 
      For 'probe' purposes, this field is ignored (except that it is copied into the first value of the returned tuple).
      
    sessions
      One or several sessions from ('first', 'second').
      If not specified, objects of all sessions are returned.
      
    expressions
      One or several expressions from ('neutral', 'smile', 'anger', 'scream').
      If not specified, objects with all expressions are returned. 
      Ignored for purpose 'enrol'.  
    
    illuminations
      One or several illuminations from ('front', 'left', 'right', 'all').
      If not specified, objects with all illuminations are returned.
      Ignored for purpose 'enrol'.  
      
    occlusions
      One or several occlusions from ('none', 'sunglasses', 'scarf').
      If not specified, objects with all occlusions are returned.
      Ignored for purpose 'enrol'.  

    genders
      One of the genders ('m', 'w') of the clients.
      If not specified, both genders are returned. 

    """
    
    # check that every parameter is as expected
    groups = self.__check_validity__(groups, "group", self.m_groups)
    self.__check_single__(protocol, "protocol", self.m_protocols)
    purposes = self.__check_validity__(purposes, "purpose", self.m_purposes)
    sessions = self.__check_validity__(sessions, "session", self.m_sessions)
    expressions = self.__check_validity__(expressions, "expression", self.m_expressions)
    illuminations = self.__check_validity__(illuminations, "illumination", self.m_illuminations)
    occlusions = self.__check_validity__(occlusions, "occlusion", self.m_occlusions)
    genders = self.__check_validity__(genders, "gender", self.m_genders)

    # assure that the given model ids are in a tuple
    if isinstance(model_ids, str) or isinstance(model_ids, unicode):
      model_ids = (model_ids,)


    def _filter_types(query):
      return query.filter(File.m_expression.in_(expressions))\
                  .filter(File.m_illumination.in_(illuminations))\
                  .filter(File.m_occlusion.in_(occlusions))\
                  .filter(File.m_session.in_(sessions))\
                  .filter(Client.m_gender.in_(genders))
      return query 


    queries = []
    probe_queries = []

    if 'world' in groups:
      queries.append(\
        _filter_types(
          self.m_session.query(File).join(Client)\
              .join((Protocol, and_(File.m_expression == Protocol.m_expression, File.m_illumination == Protocol.m_illumination, File.m_occlusion == Protocol.m_occlusion)))\
              .filter(Client.m_group == 'world')\
              .filter(Protocol.m_protocol == protocol)
        )
      )
      
    if 'dev' in groups or 'eval' in groups:
      t_groups = ('dev',) if not 'eval' in groups else ('eval',) if not 'dev' in groups else ('dev','eval')
      
      if 'enrol' in purposes:
        queries.append(\
            self.m_session.query(File).join(Client)\
                .filter(Client.m_group.in_(t_groups))\
                .filter(Client.m_gender.in_(genders))\
                .filter(File.m_purpose == 'enrol')\
        )
        
      if 'probe' in purposes:
        probe_queries.append(\
            _filter_types(
              self.m_session.query(File).join(Client)\
                  .join((Protocol, and_(File.m_expression == Protocol.m_expression, File.m_illumination == Protocol.m_illumination, File.m_occlusion == Protocol.m_occlusion)))\
                  .filter(Client.m_group.in_(t_groups))\
                  .filter(File.m_purpose == 'probe')\
                  .filter(Protocol.m_protocol == protocol)
            )
        )

    # we have collected all queries, now filter the model ids, if desired
    retval = {}
    
    for query in queries:
      # filter model ids
      if model_ids and len(model_ids) == 1:
        query = query.filter(Client.m_id.in_(model_ids))
      for k in query:
        retval[k.m_id] = (self.__make_path__(k.m_id, directory, extension), k.m_client_id, k.m_client_id, k.m_client_id, k.m_id)
          
    for query in probe_queries:
      # filter model ids
      if model_ids and len(model_ids) == 1:
        # do not filter the model id here
        for k in query:
          retval[k.m_id] = (self.__make_path__(k.m_id, directory, extension), model_ids[0], model_ids[0], k.m_client_id, k.m_id)
      else:
        # model ids are not filtered
        for k in query:
          retval[k.m_id] = (self.__make_path__(k.m_id, directory, extension), k.m_client_id, k.m_client_id, k.m_client_id, k.m_id)
        
    return retval


  def files(self, directory=None, extension=None, groups=None, protocol='all', purposes=None, model_ids=None, sessions=None, expressions=None, illuminations=None, occlusions=None, genders=None):
    """Returns a dictionary from file_ids to file paths using the specified restrictions:

    Keyword Parameters:

    directory
      A directory name that will be prepended to all file paths

    extension
      A filename extension that will be appended to all file paths

    groups
      One or several groups to which the models belong ('world', 'dev', 'eval').
    
    protocol
      One of the AR protocols ('all', 'expression', 'illumination', 'occlusion', 'occlusion_and_illumination').
      This field is ignored for group 'world'.
    
    purposes
      One or several purposes for which files should be retrieved ('enrol', 'probe').
      This field is ignored for group 'world'.

    model_ids
      If given (as a list of model id's or a single one), only the files
      belonging to the specified model id is returned. 
      For 'probe' purposes, this field is ignored (except that it is copied into the first value of the returned tuple).
      
    sessions
      One or several sessions from ('first', 'second').
      If not specified, objects of all sessions are returned.
      
    expressions
      One or several expressions from ('neutral', 'smile', 'anger', 'scream').
      If not specified, objects with all expressions are returned. 
      Ignored for purpose 'enrol'.  
    
    illuminations
      One or several illuminations from ('front', 'left', 'right', 'all').
      If not specified, objects with all illuminations are returned.
      Ignored for purpose 'enrol'.  
      
    occlusions
      One or several occlusions from ('none', 'sunglasses', 'scarf').
      If not specified, objects with all occlusions are returned.
      Ignored for purpose 'enrol'.  

    genders
      One of the genders ('m', 'w') of the clients.
      If not specified, both genders are returned. 

    """
    
    # retrieve the objects
    objects = self.objects(directory, extension, groups, protocol, purposes, model_ids, sessions, expressions, illuminations, occlusions, genders)
    
    # return the file names only
    files = {}
    for file_id, object in objects.iteritems():
      files[file_id] = object[0]
    
    return files


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

    query = self.m_session.query(File).filter(File.m_id == file_id)
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
