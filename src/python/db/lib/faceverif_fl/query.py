#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
face verification database based on file lists in the most obvious ways.
"""

from . import dbname
import os, fileinput
import re

class Database(object):
  """The dataset class opens and maintains a connection opened to the Database.

  It provides many different ways to probe for the characteristics of the data
  and for the data itself inside the database.
  """

  def __init__(self, base_dir):
    """Initialises the database with the given base directory"""
    self.base_dir = base_dir
    if not os.path.isdir(self.base_dir):
      raise RuntimeError, 'Invalid directory specified %s.' % (self.base_dir)
    self.world_subdir = 'norm'
    self.world_filename0 = 'train_world.lst'
    self.dev_subdir = 'dev'
    self.eval_subdir = 'eval'
    self.models_filename = 'for_models.lst'
    self.scores_filename = 'for_scores.lst'
    self.tnorm_filename = 'for_tnorm.lst'
    self.znorm_filename = 'for_znorm.lst'
      
  def getBaseDirectory(self):
    """Returns the base directory where the filelists defining the database
       are located."""
    return self.base_dir

  def setBaseDirectory(self, base_dir):
    """Resets the base directory where the filelists defining the database
       are located."""
    self.base_dir = base_dir
    if not os.path.isdir(self.base_dir):
      raise RuntimeError, 'Invalid directory specified %s.' % (self.base_dir)

  def __check_validity__(self, l, obj, valid):
    """Checks validity of user input data against a set of valid values"""
    if not l: return valid
    elif isinstance(l, str): return self.__check_validity__((l,), obj, valid)
    for k in l:
      if k not in valid:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
    return l

  def __append_model_ids__(self, ids_list, filename):
    """Appends the model_ids contained in the given filename into the given 
       list"""
    if os.path.isfile(filename): 
      try: 
        for line in fileinput.input(filename):
          model_id = re.findall('[\w/]+', line)[1]
          if not model_id in ids_list:
            ids_list.append(model_id)  
      except IOError as e:
        raise RuntimeError, 'Error reading the file %s.' % (filename,)
    else:
      raise RuntimeError, 'File %s does not exist.' % (filename,)

  def __make_path__(self, stem, directory, extension):
    if not extension: extension = ''
    if directory: return os.path.join(directory, stem + extension)
    return stem + extension


  def __append_objects__(self, objects_list, model_ids, filename, directory, extension, classes=None):
    """Appends the files contained in the given filename into the given 
       list"""
    if os.path.isfile(filename): 
      try: 
        for line in fileinput.input(filename):
          parsed_list = re.findall('[\w/]+', line)
          sfile = parsed_list[0]
          model_id = parsed_list[1]
          if len(parsed_list)>2: claimed_id = parsed_list[2] 
          else: claimed_id = model_id
          if len(parsed_list)>3: real_id = parsed_list[3] 
          else: real_id = claimed_id
          if (not model_ids) or model_id in model_ids:
            if(not classes \
                or ('client' in classes and claimed_id==real_id) \
                or ('impostor' in classes and claimed_id!=real_id) ):
              objects_list.append( (self.__make_path__(sfile, directory, extension), model_id, claimed_id, real_id, sfile))
      except IOError as e:
        raise RuntimeError, 'Error reading the file %s.' % (filename,)
    else:
      raise RuntimeError, 'File %s does not exist.' % (filename,)

  def __append_files__(self, files_list, model_ids, filename, directory, extension):
    """Appends the files contained in the given filename into the given 
       list"""
    if os.path.isfile(filename): 
      try: 
        for line in fileinput.input(filename):
          parsed_list = re.findall('[\w/]+', line)
          sfile = parsed_list[0]
          model_id = parsed_list[1]
          if model_id in model_ids:
            files_list.append(self.__make_path__(sfile, directory, extension))
      except IOError as e:
        raise RuntimeError, 'Error reading the file %s.' % (filename,)
    else:
      raise RuntimeError, 'File %s does not exist.' % (filename,)


  def models(self, protocol=None, groups=None, subworld=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    groups
      The groups to which the clients belong ("dev", "eval", "world").

    subworld
      Specify a split of the world data ("")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    Returns: A list containing all the client ids which have the given
    properties.
    """

    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_SUBWORLDS = ('',)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    subworld = self.__check_validity__(subworld, "subworld", VALID_SUBWORLDS)

    retval = []
    # List of the models in the world set
    if "world" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.world_subdir, self.world_filename0) )

    # List of the models in the dev set
    if "dev" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.dev_subdir, self.models_filename) )

     # List of the models in the eval set
    if "eval" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.eval_subdir, self.models_filename) )

    return retval

  def Tmodels(self, protocol=None, groups=None):
    """Returns a set of T-Norm clients for the specific query by the user.

    Keyword Parameters:

    groups
      The groups to which the clients belong ("dev", "eval").

    Returns: A list containing all the model ids belonging to the given group.
    """

    VALID_GROUPS = ('dev', 'eval')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)

    retval = []
    # List of the T-Norm models in the dev set
    if "dev" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.dev_subdir, self.tnorm_filename))

    # List of the T-Norm models in the eval set
    if "eval" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.eval_subdir, self.tnorm_filename))

    return retval



  def Zmodels(self, protocol=None, groups=None):
    """Returns a set of Z-Norm clients for the specific query by the user.

    Keyword Parameters:

    groups
      The groups to which the clients belong ("dev", "eval").

    Returns: A list containing all the model ids belonging to the given group.
    """

    VALID_GROUPS = ('dev', 'eval')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)

    retval = []
    # List of the Z-Norm models in the dev set
    if "dev" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.dev_subdir, self.znorm_filename))

    # List of the Z-Norm models in the eval set
    if "eval" in groups:
      self.__append_model_ids__(retval, os.path.join(self.base_dir, self.eval_subdir, self.znorm_filename))

    return retval


  def objects(self, directory=None, extension=None, protocol=None, purposes=None, 
      model_ids=None, groups=None, classes=None, subworld=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    purposes
      The purposes required to be retrieved ("enrol", "probe") or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values. This field is
      ignored for the data from the "world" group.

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ("dev", "eval", "world") or a tuple with several of them.
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    subworld
      Specify a split of the world data ("")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    Returns: A list containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. 
    """

    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_CLASSES = ('client', 'impostor')
    VALID_SUBWORLDS = ('',)

    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)
    subworld = self.__check_validity__(subworld, "subworld", VALID_SUBWORLDS)

    retval = []

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
    
    if 'world' in groups:
      self.__append_objects__(retval, model_ids, \
        os.path.join(self.base_dir, self.world_subdir, self.world_filename0), directory, extension)

    if 'dev' in groups:
      if('enrol' in purposes):
        self.__append_objects__(retval, model_ids, \
          os.path.join(self.base_dir, self.dev_subdir, self.models_filename), directory, extension)
      if('probe' in purposes):
        if('client' in classes):
          self.__append_objects__(retval, model_ids, \
            os.path.join(self.base_dir, self.dev_subdir, self.scores_filename), directory, extension, classes=('client',))
        if('impostor' in classes):
          self.__append_objects__(retval, model_ids, \
            os.path.join(self.base_dir, self.dev_subdir, self.scores_filename), directory, extension, classes=('impostor',))

    if 'eval' in groups:
      if('enrol' in purposes):
        self.__append_objects__(retval, model_ids, \
          os.path.join(self.base_dir, self.eval_subdir, self.models_filename), directory, extension)
      if('probe' in purposes):
        if('client' in classes):
          self.__append_objects__(retval, model_ids, \
            os.path.join(self.base_dir, self.eval_subdir, self.scores_filename), directory, extension, classes=('client',))
        if('impostor' in classes):
          self.__append_objects__(retval, model_ids, \
            os.path.join(self.base_dir, self.eval_subdir, self.scores_filename), directory, extension, classes=('impostor',))

    return retval

  def files(self, directory=None, extension=None, protocol=None, purposes=None,
      model_ids=None, groups=None, classes=None, subworld=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    purposes
      The purposes required to be retrieved ("enrol", "probe") or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values. This field is
      ignored for the data from the "world" group.

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ("dev", "eval", "world") or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    subworld
      Specify a split of the world data ("")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. Clients from other groups ("dev", "eval")
      will in this case be ignored.

    Returns: A list containing the resolved filenames considering all
    the filtering criteria. 
    """

    retval = {}
    d = self.objects(directory, extension, purposes, model_ids, groups, classes, subworld)
    for k in d: retval[k] = d[k][0]

    return retval


  def Tobjects(self, directory=None, extension=None, protocol=None, model_ids=None, groups=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("dev", "eval").

    Returns: A list containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. 
    """

    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('dev', 'eval')
    VALID_CLASSES = ('client', 'impostor')

    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)

    retval = []

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)

    # Please note that T-Norm file lists only contain data for enrolling the T-Norm models
    # (i.e. classes == 'client')
    if 'dev' in groups:
      self.__append_objects__(retval, model_ids, \
        os.path.join(self.base_dir, self.dev_subdir, self.tnorm_filename), \
        directory, extension, classes=('client',))

    if 'eval' in groups:
      self.__append_objects__(retval, model_ids, \
        os.path.join(self.base_dir, self.eval_subdir, self.tnorm_filename), \
        directory, extension, classes=('client',))

    return retval


  def Tfiles(self, directory=None, extension=None, protocol=None, model_ids=None, groups=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("dev", "eval").

    Returns: A list containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. 
    """

    retval = {}
    d = self.Tobjects(directory, extension, protocol, model_ids, groups)
    for k in d: retval[k] = d[k][0]

    return retval


  def Zobjects(self, directory=None, extension=None, protocol=None, model_ids=None, groups=None):
    """Returns a set of filenames to perform Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("dev", "eval").

    Returns: A list containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. 
    """

    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('dev', 'eval')
    VALID_CLASSES = ('client', 'impostor')

    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)

    retval = []

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
    
    # Please note that Z-Norm file lists only contain impostor accesses
    # (i.e. classes == 'impostor')
    if 'dev' in groups:
      self.__append_objects__(retval, model_ids, \
        os.path.join(self.base_dir, self.dev_subdir, self.znorm_filename), \
        directory, extension, client=('impostor',))

    if 'eval' in groups:
      self.__append_objects__(retval, model_ids, \
        os.path.join(self.base_dir, self.eval_subdir, self.tnorm_filename), \
        directory, extension, client=('impostor',))

    return retval

  def Zfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None):
    """Returns a set of filenames to perform Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("dev", "eval").

    Returns: A list containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria.
    """

    retval = {}
    d = self.Zobjects(directory, extension, protocol, model_ids, groups)
    for k in d: retval[k] = d[k][0]

    return retval
