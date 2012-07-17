#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Fri Jul  6 16:45:41 CEST 2012
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
FRGC database in the most obvious ways.
"""

from .models import get_list, get_mask, get_positions, client_from_file, client_from_model

import os

class Database(object):
  """The Database class reads the original XML lists and provides access 
  using the common bob.db API.
  """

  def __init__(self, base_dir = '/idiap/resource/database/frgc/FRGC-2.0-dist'):
    # opens a session to the database - keep it open until the end
    self.m_base_dir = base_dir
    self.m_groups  = ('world', 'dev')
    self.m_purposes = ('enrol', 'probe')
    self.m_protocols = ('2.0.1', '2.0.2', '2.0.4') # other protocols might be supported later.
    self.m_mask_types = ('maskI', 'maskII', 'maskIII') # usually, only maskIII (the most difficult one) is used.

  def __check_validity__(self, elements, description, possibilities, default=None):
    """Checks validity of user input data against a set of valid values"""
    if not elements: 
      return default if default else possibilities 
    if not isinstance(elements, list) and not isinstance(elements, tuple): 
      return self.__check_validity__((elements,), description, possibilities, default)
    for k in elements:
      if k not in possibilities:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (description, k, possibilities)
    return elements
  
  def __check_single__(self, element, description, possibilities):
    """Checks validity of user input data against a set of valid values"""
    if not element:
      raise RuntimeError, 'Please select one element from %s for %s' % (possibilities, description)
    if isinstance(element,tuple) or isinstance (element,list):
      if len(element) > 1:
        raise RuntimeError, 'For %s, only single elements from %s are allowed' % (description, possibilities)
      element = element[0]
    if element not in possibilities:
      raise RuntimeError, 'The given %s "%s" is not allowed. Please choose one of %s' % (description, element, possibilities)

  def __make_path__(self, stem, directory, extension, replacement=None):
    """Generates the file name for the given file name. 
    If directory and extension '.jpg' are specified,
    extensions are automatically replaced by '.JPG' if necessary."""
    if not extension: 
      extension = ''
    if replacement and extension == '.jpg':
      extension = replacement
    if directory: 
      return os.path.join(directory, stem + extension)
    return stem + extension
    
    if directory:
      if not extension: 
        return os.path.join(directory, stem)
      full_path = os.path.join(directory, stem + extension)
      if extension == '.jpg' and not os.path.exists(full_path):
        capital_path = os.path.join(directory, stem + '.JPG')
        if os.path.exists(capital_path):
          return capital_path
      return full_path

    if not extension:
      return stem        
    return stem + extension
    

  def clients(self, groups=None, protocol=None, purposes=None, mask_type='maskIII'):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    groups
      One or several groups to which the models belong ('world', 'dev').

    protocol
      One or several of the GBU protocols ('2.0.1', '2.0.2, '2.0.4'), 
      required only if one of the groups is 'dev'.
      
    purposes
      One or several groups for which files should be retrieved ('enrol', 'probe').
      Only used when the group is 'dev'· 
      For some protocol/mask_type pairs, not all clients are used for enrollment / for probe.

    mask_type
      One of the mask types ('maskI', 'maskII', 'maskIII').
    
    Returns: A list containing all the client id's which have the given properties.
    """

    groups = self.__check_validity__(groups, "group", self.m_groups)

    retval = set()

    if 'world' in groups:
      for file in get_list(self.m_base_dir, 'world', protocol):
        retval.add(file.m_client_id)
        
    if 'dev' in groups:
      # validity checks
      purposes = self.__check_validity__(purposes, "purpose", self.m_purposes)
      self.__check_single__(protocol, "protocol", self.m_protocols)
      self.__check_single__(mask_type, "mask type", self.m_mask_types)

      # take only those models/probes that are really required by the current mask
      mask = get_mask(self.m_base_dir, protocol, mask_type)

      if 'enrol' in purposes:
        files = get_list(self.m_base_dir, 'dev', protocol, purpose='enrol')
        for index in range(len(files)):
          # check if this model is used by the mask
          if (mask[:,index] > 0).any():
            retval.add(files[index].m_client_id)

      if 'probe' in purposes:
        files = get_list(self.m_base_dir, 'dev', protocol, purpose='probe')
        for index in range(len(files)):
          # check if this probe is used by the mask
          if (mask[index,:] > 0).any():
            retval.add(files[index].m_client_id)
      
    
    return sorted(list(retval))


  def models(self, groups=None, protocol=None, mask_type='maskIII'):
    """Returns a set of models for the specific query by the user.
    
    The models are dependent on the protocol and the mask. 
    Only those FRGC "target" files are returned that are required by the given mask!  
    
    .. warning :: 
      Clients, models, and files are not identical for the FRGC database!
      Model ids are neither client nor file id's, so please do not mix that up!

    Keyword Parameters:
    
    groups
      One or several groups to which the models belong ('world', 'dev').
    
    protocol
      One or several of the GBU protocols ('2.0.1', '2.0.2, '2.0.4'), 
      required only if one of the groups is 'dev'.
      
    mask_type
      One of the mask types ('maskI', 'maskII', 'maskIII').
    
    Returns: A list containing all the model id's belonging to the given group.
    """

    groups = self.__check_validity__(groups, "group", self.m_groups)
    # for models, purpose is always 'enrol'
    purpose = 'enrol'

    retval = set()
    if 'world' in groups:
      for file in get_list(self.m_base_dir, 'world'):
        retval.add(file.m_model)
        
    if 'dev' in groups:
      self.__check_single__(protocol, "protocol", self.m_protocols)
      self.__check_single__(mask_type, "mask type", self.m_mask_types)
      files = get_list(self.m_base_dir, 'dev', protocol, purpose)
      # take only those models that are really required by the current mask
      mask = get_mask(self.m_base_dir, protocol, mask_type)
      for index in range(len(files)):
        if (mask[:,index] > 0).any():
          retval.add(files[index].m_model)
        
    return retval
      

  def get_client_id_from_model_id(self, model_id):
    """Returns the client_id attached to the given model_id.
    
    Keyword Parameters:

    model_id
      The model_id to consider
      
      .. warning ::
        The given model_id must have been the result of a previous call (models(), files()) 
        to the SAME database object, otherwise it will not be known or might be corrupted.
      
    Returns: The client_id attached to the given model_id
    """
   
    return client_from_model(model_id)


  def get_client_id_from_file_id(self, file_id):
    """Returns the client_id (real client id) attached to the given file_id
    
    Keyword Parameters:

    file_id
      The file_id to consider

      .. warning ::
        The given client_id must have been the result of a previous call (clients(), files()) 
        to the SAME database object, otherwise it will not be known.
      

    Returns: The client_id attached to the given file_id
    """

    return client_from_file(file_id)


  def objects(self, directory=None, extension=None, groups=None, protocol=None, purposes=None, model_ids=None, mask_type='maskIII'):
    """Using the specified restrictions, this function returns a dictionary from file_ids to a tuple containing:
    
    * 0: the resolved filename
    * 1: the model id (if one (and only one) model_id is given, it is copied here, otherwise the model id of the file)
    * 2: the claimed client id attached to the model (in case of the AR database: identical to 1)
    * 3: the real client id (for a probe image, the client id of the probe, otherwise identical to 1)
    * 4: the "stem" path (basename of the file; in case of the AR database: identical to the file id)

    Keyword Parameters:

    directory
      A directory name that will be prepended to all file paths.

    extension
      A filename extension that will be appended to all file paths.
      If the extension is '.jpg', but the expected extension is '.JPG',
      the extension is automatically corrected.

    groups
      One or several groups to which the models belong ('world', 'dev').
      'world' files are "Training", whereas 'dev' files are "Target" and/or "Query".
    
    protocol
      One of the FRGC protocols ('2.0.1', '2.0.2', '2.0.4').
      Needs to be specified, when 'dev' is amongst the groups.

    purposes
      One or several groups for which files should be retrieved ('enrol', 'probe').
      Only used when the group is 'dev'· 
      In FRGC terms, 'enrol' is "Target", while 'probe' is "Target" (protocols '2.0.1' and '2.0.2') or "Query" (protocol '2.0.4')

    model_ids
      If given (as a list of model id's or a single one), only the files
      belonging to the specified model id is returned.

      .. warning :: 
        When querying objects of group 'world', model ids are expected to be client ids (returned by 'clients()'), 
        whereas for group 'dev' model ids are real model ids (as returned by 'models()')

    mask_type
      One of the mask types ('maskI', 'maskII', 'maskIII').

    """
    
    # check that every parameter is as expected
    groups = self.__check_validity__(groups, "group", self.m_groups)

    if isinstance(model_ids, str):
      model_ids = (model_ids,)
    
    retval = {}
    
    if 'world' in groups:
      # extract training files
      for file in get_list(self.m_base_dir, 'world'):
        if not model_ids or file.m_client_id in model_ids:
          for id, path in file.m_files.items():
            retval[id] = (self.__make_path__(path, directory, extension, file.m_extensions[id]), file.m_model, file.m_client_id, file.m_client_id, path)
      
    if 'dev' in groups:
      # check protocol, mask, and purposes only in group dev
      self.__check_single__(protocol, "protocol", self.m_protocols)
      self.__check_single__(mask_type, "mask type", self.m_mask_types)
      purposes = self.__check_validity__(purposes, "purpose", self.m_purposes)
      
      # extract dev files
      if 'enrol' in purposes:
        model_files = get_list(self.m_base_dir, 'dev', protocol, 'enrol')
        # return only those files that are required by the given protocol 
        mask = get_mask(self.m_base_dir, protocol, mask_type)
        for model_index in range(len(model_files)):
          model = model_files[model_index]
          if not model_ids or model.m_model in model_ids:
            # test if the model is used by this mask 
            if (mask[:,model_index] > 0).any():
              for id, path in model.m_files.items():
                retval[id] = (self.__make_path__(path, directory, extension, model.m_extensions[id]), model.m_model, model.m_client_id, model.m_client_id, path)
            
      if 'probe' in purposes:
        probe_files = get_list(self.m_base_dir, 'dev', protocol, 'probe')
        
        if model_ids:
          # select only that files that belong to the models of with the given ids
          model_files = get_list(self.m_base_dir, 'dev', protocol, 'enrol')
          mask = get_mask(self.m_base_dir, protocol, mask_type)
          
          for model_index in range(len(model_files)):
            model = model_files[model_index]
            if model.m_model in model_ids:
              for probe_index in range(len(probe_files)):
                if mask[probe_index, model_index]:
                  probe = probe_files[probe_index]
                  for id, path in probe.m_files.items():
                    retval[id] = (self.__make_path__(path, directory, extension, probe.m_extensions[id]), model.m_model , model.m_client_id, probe.m_client_id, path)
          
        else: # no model_ids
          # simply get all the probe files
          for file in probe_files:
            for id, path in file.m_files.items():
              retval[id] = (self.__make_path__(path, directory, extension, file.m_extensions[id]), file.m_model, file.m_client_id, file.m_client_id, path)
          
    return retval


  def files(self, directory=None, extension=None, groups=None, protocol=None, purposes=None, model_ids=None, mask_type='maskIII'):
    """Returns a dictionary from file_ids to file paths using the specified restrictions:

    Keyword Parameters:

    directory
      A directory name that will be prepended to all file paths.

    extension
      A filename extension that will be appended to all file paths.
      If the extension is '.jpg', but the expected extension is '.JPG',
      the extension is automatically corrected.

    groups
      One or several groups to which the models belong ('world', 'dev').
      'world' files are "Training", whereas 'dev' files are "Target" and/or "Query".
    
    protocol
      One of the FRGC protocols ('2.0.1', '2.0.2', '2.0.4').
      Needs to be specified, when 'dev' is amongst the groups.
    
    purposes
      One or several groups for which files should be retrieved ('enrol', 'probe').
      Only used when the group is 'dev'· 
      In FRGC terms, 'enrol' is "Target", while 'probe' is "Target" (protocols '2.0.1' and '2.0.2') or "Query" (protocol '2.0.4')

    model_ids
      If given (as a list of model id's or a single one), only the files
      belonging to the specified model id is returned.
      
      .. warning :: 
        When querying objects of group 'world', model ids are expected to be client ids (returned by 'clients()'), 
        whereas for group 'dev' model ids are real model ids (as returned by 'models()')
      
    mask_type
      One of the mask types ('maskI', 'maskII', 'maskIII').

    """
    
    # retrieve the objects
    objects = self.objects(directory, extension, groups, protocol, purposes, model_ids, mask_type)
    # return the file names only
    files = {}
    for file_id, object in objects.iteritems():
      files[file_id] = object[0]
    
    return files
  

  def positions(self, directory=None, extension=None, groups=None, protocols=None, purposes=None):
    """Returns a list of 8 integrals (positions of right eye (re), left eye (le), mouth (m), nose (n)) for the given query.
    
    Returns:
    
      A dictionary with the file id as key and:
      
      * 1: The resolved file name
      * 2: The positions in the order (re_x, re_y, le_x, le_y, m_x, m_y, n_x, n_y) 
    
    Keyword Parameters:

    directory
      A directory name that will be prepended to all file paths

    extension
      A filename extension that will be appended to all file paths

    groups
      One or several groups to which the models belong ('world', 'dev').
      'world' files are "Training", whereas 'dev' files are "Target" and/or "Query".
    
    protocols
      One or more of the FRGC protocols ('2.0.1', '2.0.2', '2.0.4').
      If not specified, all protocols are considered. 
      Ignored for the 'world' group.
    
    purposes
      One or several groups for which files should be retrieved ('enrol', 'probe').
      Only used when the group is 'dev'· 
      Here, 'enrol' is "Target", while probe is "Target" (protocols '2.0.1' and '2.0.2') or "Query" (protocol '2.0.4')

    """

    # check that every parameter is as expected
    groups = self.__check_validity__(groups, "group", self.m_groups)
    protocols = self.__check_validity__(protocols, "protocol", self.m_protocols)
    purposes = self.__check_validity__(purposes, "purpose", self.m_purposes)

    positions = {}

    for protocol in protocols:    
      if 'world' in groups:
        for file in get_list(self.m_base_dir, 'world'):
          for id, path in file.m_files.items():
            positions[id] = (self.__make_path__(path, directory, extension), get_positions(self.m_base_dir, id))

      if 'dev' in groups:
        for purpose in purposes:
          for file in get_list(self.m_base_dir, 'dev', protocol, purpose):
            for id, path in file.m_files.items():
              positions[id] = (self.__make_path__(path, directory, extension), get_positions(self.m_base_dir, id))
        
    return positions


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

    # The implementation of this function could be ugly.
    # Hence, I don't do it. Ask me if you need this function to be implemented.
    raise NotImplementedError("Implement me!")

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

    # The implementation of this function could be ugly.
    # Hence, I don't do it. Ask me if you need this function to be implemented.
    raise NotImplementedError("Implement me!")
