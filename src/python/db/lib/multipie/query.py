#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
Multi-PIE database in the most obvious ways.
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
    self.session = utils.session(dbname())

  def __check_validity__(self, l, obj, valid):
    """Checks validity of user input data against a set of valid values"""
    if not l: return valid
    elif isinstance(l, str): return self.__check_validity__((l,), obj, valid)
    for k in l:
      if k not in valid:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
    return l

  def clients(self, protocol=None, groups=None, gender=None, birthyear=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('M', 'U', 'G')

    groups
      The groups to which the clients belong ('dev', 'eval', 'world')

    gender
      The genders to which the clients belong ('f', 'm')

    birthyear
      The birth year of the clients (in the range [1900,2050])

    Returns: A list containing all the client ids which have the given
    properties.
    """

    VALID_PROTOCOLS = ('M', 'U', 'G')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_GENDERS = ('m', 'f')
    VALID_BIRTHYEARS = range(1900, 2050)
    VALID_BIRTHYEARS.append(57) # bug in subject_list.txt (57 instead of 1957)
    protocol = self.__check_validity__(protocol, 'protocol', VALID_PROTOCOLS)
    groups = self.__check_validity__(groups, 'group', VALID_GROUPS)
    gender = self.__check_validity__(gender, 'gender', VALID_GENDERS)
    birthyear = self.__check_validity__(birthyear, 'birthyear', VALID_BIRTHYEARS)
    # List of the clients
    q = self.session.query(Client).\
          filter(Client.sgroup.in_(groups)).\
          filter(Client.gender.in_(gender)).\
          filter(Client.birthyear.in_(birthyear)).\
          order_by(Client.id)
    retval = []
    for id in [k.id for k in q]: 
      retval.append(id)
    return retval

  def Tclients(self, protocol=None, groups=None):
    """Returns a set of T-Norm clients for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('M', 'U', 'G')
    
    groups
      The groups to which the clients belong ('dev', 'eval').

    Returns: A list containing all the client ids belonging to the given group.
    """

    VALID_GROUPS = ('dev', 'eval')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    tgroups = []
    if 'dev' in groups:
      tgroups.append('eval')
    if 'eval' in groups:
      tgroups.append('dev')
    return self.clients(protocol, tgroups)

  def Zclients(self, protocol=None, groups=None):
    """Returns a set of Z-Norm clients for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('M', 'U', 'G')
    
    groups
      The groups to which the clients belong ('dev', 'eval').

    Returns: A list containing all the client ids belonging to the given group.
    """

    VALID_GROUPS = ('dev', 'eval')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    zgroups = []
    if 'dev' in groups:
      zgroups.append('eval')
    if 'eval' in groups:
      zgroups.append('dev')
    return self.clients(protocol, zgroups)

  def models(self, protocol=None, groups=None):
    """Returns a set of models for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('M', 'U', 'G')
    
    groups
      The groups to which the subjects attached to the models belong ('dev', 'eval', 'world')

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.clients(protocol, groups)

  def Tmodels(self, protocol=None, groups=None):
    """Returns a set of T-Norm models for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('M', 'U', 'G')
    
    groups
      The groups to which the models belong ('dev', 'eval').

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.Tclients(protocol, groups)

  def Zmodels(self, protocol=None, groups=None):
    """Returns a set of Z-Norm models for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('M', 'U', 'G')
    
    groups
      The groups to which the models belong ('dev', 'eval').

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.Zclients(protocol, groups)


  def getClientIdFromModelId(self, model_id):
    """Returns the client_id attached to the given model_id
    
    Keyword Parameters:

    model_id
      The model_id to consider

    Returns: The client_id attached to the given model_id
    """
    return model_id

  def getClientIdFromTmodelId(self, model_id):
    """Returns the client_id attached to the given T-Norm model_id
    
    Keyword Parameters:

    model_id
      The model_id to consider

    Returns: The client_id attached to the given T-Norm model_id
    """
    return model_id

  def getClientIdFromFileId(self, file_id):
    """Returns the client_id (real client id) attached to the given file_id
    
    Keyword Parameters:

    file_id
      The file_id to consider

    Returns: The client_id attached to the given file_id
    """
    q = self.session.query(File).\
          filter(File.id == file_id)
    if q.count() !=1:
      #throw exception?
      return None
    else:
      return q.first().client_id

  def getInternalPathFromFileId(self, file_id):
    """Returns the unique "internal path" attached to the given file_id
    
    Keyword Parameters:

    file_id
      The file_id to consider

    Returns: The internal path attached to the given file_id
    """
    q = self.session.query(File).\
          filter(File.id == file_id)
    if q.count() !=1:
      #throw exception?
      return None
    else:
      return q.first().path


  def objects(self, directory=None, extension=None, protocol=None,
      purposes=None, model_ids=None, groups=None, classes=None, subworld=None,
      expressions=None, world_sampling=1, world_noflash=False, world_first=False):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Multi-PIE protocols ('M', 'U', 'G').

    purposes
      The purposes required to be retrieved ('enrol', 'probe') or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values. This field is
      ignored for the data from the "world" group.

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ('dev', 'eval', 'world') or a tuple with several of
      them.  If 'None' is given (this is the default), it is considered the
      same as a tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.
  
    subworld
      if only a subset of the world data should be used

    expressions
      The (face) expressions to be retrieved ('neutral', 'smile', 'surprise',
      'squint', 'disgust', 'scream') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as 
      a tuple with all possible values.

    world_sampling
      Samples the files from the world data set. Keeps only files such as:
      File.client_id + File.shot_id % world_sampling == 0. This argument should
      be an integer between 1 (keep everything) and 19.  It is not used if
      world_noflash is also set.

    world_noflash
      Keeps the files from the world dataset recorded without flash (shot 1)
      
    world_first
      Only uses data from the first recorded session of each user of the world
      dataset.

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the Multi-PIE database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('M', 'U', 'G')
    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_CLASSES = ('client', 'impostor')
    VALID_EXPRESSIONS = ('neutral', 'smile', 'surprise', 'squint', 'disgust', 'scream')

    protocol = self.__check_validity__(protocol, 'protocol', VALID_PROTOCOLS)
    purposes = self.__check_validity__(purposes, 'purpose', VALID_PURPOSES)
    groups = self.__check_validity__(groups, 'group', VALID_GROUPS)
    classes = self.__check_validity__(classes, 'class', VALID_CLASSES)
    expressions = self.__check_validity__(expressions, 'expression', VALID_EXPRESSIONS)

    retval = {}
    
    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
   
    if 'world' in groups:
      # Multiview
      q = self.session.query(File,Expression).join(Client).join(FileMultiview).\
            filter(Client.sgroup == 'world').\
            filter(Expression.name.in_(expressions)).\
            filter(and_(File.img_type == 'multiview', File.session_id == Expression.session_id,\
                        File.recording_id == Expression.recording_id, FileMultiview.shot_id != 19))
      if model_ids:
        q = q.filter(File.client_id.in_(model_ids))
      if( world_sampling != 1 and world_noflash == False):
        q = q.filter(((File.client_id + FileMultiview.shot_id) % world_sampling) == 0)
      if( world_noflash == True):
        q = q.filter(FileMultiview.shot_id == 0)
      if( world_first == True):
        q = q.filter(File.session_id == Client.first_session)
      q = q.order_by(File.client_id, File.session_id, FileMultiview.shot_id)
      for k in q:
        retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)
    
      # Highres
      # TODO

    if('dev' in groups or 'eval' in groups):
      # Multiview
      if('enrol' in purposes):
        q = self.session.query(File, Protocol, ProtocolName, ProtocolMultiview).join(Client).join(FileMultiview).\
              filter(and_(Client.sgroup.in_(groups), Client.sgroup != 'world')).\
              filter(and_(ProtocolName.name.in_(protocol), Protocol.name == ProtocolName.name, Protocol.sgroup.in_(groups),\
                          Protocol.sgroup != 'world', Protocol.img_type == 'multiview', Protocol.session_id == File.session_id,\
                          Protocol.recording_id == File.recording_id, Protocol.purpose == 'enrol')).\
              filter(and_(Protocol.id == ProtocolMultiview.id, ProtocolMultiview.camera_id == FileMultiview.camera_id,\
                          ProtocolMultiview.shot_id == FileMultiview.shot_id))
        if model_ids:
          q = q.filter(and_(Client.id.in_(model_ids)))
        q = q.order_by(File.client_id, File.session_id, FileMultiview.shot_id)
        for k in q:
          retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)

      if('probe' in purposes):
        if('client' in classes):
          q = self.session.query(File, Protocol, ProtocolName, ProtocolMultiview).join(Client).join(FileMultiview).\
                filter(and_(Client.sgroup.in_(groups), Client.sgroup != 'world')).\
                filter(and_(ProtocolName.name.in_(protocol), Protocol.name == ProtocolName.name, Protocol.sgroup.in_(groups),\
                            Protocol.sgroup != 'world', Protocol.img_type == 'multiview', Protocol.session_id == File.session_id,\
                            Protocol.recording_id == File.recording_id, Protocol.purpose == 'probe')).\
                filter(and_(Protocol.id == ProtocolMultiview.id, ProtocolMultiview.camera_id == FileMultiview.camera_id,\
                            ProtocolMultiview.shot_id == FileMultiview.shot_id))
          if model_ids:
            q = q.filter(Client.id.in_(model_ids))
          q = q.order_by(File.client_id, File.session_id, FileMultiview.shot_id)
          for k in q:
            retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)
        if('impostor' in classes):
          q = self.session.query(File, Protocol, ProtocolName, ProtocolMultiview).join(Client).join(FileMultiview).\
                filter(and_(Client.sgroup.in_(groups), Client.sgroup != 'world')).\
                filter(and_(ProtocolName.name.in_(protocol), Protocol.name == ProtocolName.name, Protocol.sgroup.in_(groups),\
                            Protocol.sgroup != 'world', Protocol.img_type == 'multiview', Protocol.session_id == File.session_id,\
                            Protocol.recording_id == File.recording_id, Protocol.purpose == 'probe')).\
                filter(and_(Protocol.id == ProtocolMultiview.id, ProtocolMultiview.camera_id == FileMultiview.camera_id,\
                            ProtocolMultiview.shot_id == FileMultiview.shot_id))
          if(model_ids and len(model_ids)==1):
            q = q.filter(not_(Client.id.in_(model_ids)))
          q = q.order_by(File.client_id, File.session_id, FileMultiview.shot_id)
          for k in q:
            if(model_ids and len(model_ids) == 1):
              retval[k[0].id] = (make_path(k[0].path, directory, extension), model_ids[0], model_ids[0], k[0].client_id, k[0].path)
            else:
              retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)

      # Highres
      # TODO

    return retval

  def files(self, directory=None, extension=None, protocol=None,
      purposes=None, model_ids=None, groups=None, classes=None, subworld=None,
      expressions=None, world_sampling=1, world_noflash=False, world_first=False):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Multi-PIE protocols ('M', 'U', 'G').

    purposes
      The purposes required to be retrieved ('enrol', 'probe') or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values. This field is
      ignored for the data from the 'world' group.

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ('dev', 'eval', 'world') or a tuple with several of
      them.  If 'None' is given (this is the default), it is considered the
      same as a tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    expressions
      The (face) expressions to be retrieved ('neutral', 'smile', 'surprise',
      'squint', 'disgust', 'scream') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as 
      a tuple with all possible values.

    world_sampling
      Samples the files from the world data set. Keeps only files such as:
      File.client_id + File.shot_id % world_sampling == 0. This argument should
      be an integer between 1 (keep everything) and 20.  It is not used if
      world_noflash is also set.

    world_noflash
      Keeps the files from the world dataset recorded without flash (shots 1
      and 19)
 
    world_first Only uses data from the first recorded session of each user of
    the world dataset.

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the Multi-PIE database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    retval = {}
    d = self.objects(directory, extension, protocol, purposes, model_ids, groups, classes, subworld, expressions, world_sampling, world_noflash, world_first)
    for k in d: retval[k] = d[k][0]

    return retval


  def Tobjects(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, expressions=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Multi-PIE protocols ('M', 'U', 'G').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ('dev', 'eval').

    expressions
      The (face) expressions to be retrieved ('neutral', 'smile', 'surprise',
      'squint', 'disgust', 'scream') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as 
      a tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)

    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the Multi-PIE database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    VALID_GROUPS = ('dev', 'eval')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    tgroups = []
    if 'dev' in groups:
      tgroups.append('eval')
    if 'eval' in groups:
      tgroups.append('dev')
    return self.objects(directory, extension, protocol, 'enrol', model_ids, tgroups, 'client', None, expressions)

  def Tfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, expressions=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Multi-PIE protocols ('M', 'U', 'G').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ('dev', 'eval').

    expressions
      The (face) expressions to be retrieved ('neutral', 'smile', 'surprise',
      'squint', 'disgust', 'scream') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as 
      a tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)

    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the Multi-PIE database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.Tobjects(directory, extension, protocol, model_ids, groups, expressions)
    for k in d: retval[k] = d[k][0]

    return retval

  def Zobjects(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, expressions=None):
    """Returns a set of filenames of impostors for Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Multi-PIE protocols ('M', 'U', 'G').

    model_ids
      Only retrieves the files for the provided list of model ids (client id).  
      If 'None' is given (this is the default), no filter over the model_ids 
      is performed.

    groups
      The groups to which the clients belong ('dev', 'eval').

    expressions
      The (face) expressions to be retrieved ('neutral', 'smile', 'surprise',
      'squint', 'disgust', 'scream') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as 
      a tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the client id
      - 2: the "stem" path (basename of the file)

    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the Multi-PIE database. Conserve these
    numbers if you wish to save processing results later on.
    """

    VALID_GROUPS = ('dev', 'eval')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)

    zgroups = []
    if 'dev' in groups:
      zgroups.append('eval')
    if 'eval' in groups:
      zgroups.append('dev')

    retval = {}
    d = self.objects(directory, extension, protocol, 'probe', model_ids, zgroups, 'client', None, expressions)
    for k in d: retval[k] = d[k]

    return retval

  def Zfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, expressions=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Multi-PIE protocols ('M', 'U', 'G').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ('dev', 'eval').

    expressions
      The (face) expressions to be retrieved ('neutral', 'smile', 'surprise',
      'squint', 'disgust', 'scream') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as 
      a tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the client id
      - 2: the "stem" path (basename of the file)

    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the Multi-PIE database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.Zobjects(directory, extension, protocol, model_ids, groups, expressions)
    for k in d: retval[k] = d[k][0]

    return retval


  def save_one(self, id, obj, directory, extension):
    """Saves a single object supporting the bob save() protocol.

    This method will call save() on the the given object using the correct
    database filename stem for the given id.
    
    Keyword Parameters:

    id
      The id of the object in the database table "file".

    obj
      The object that needs to be saved, respecting the bob save() protocol.

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs()

    extension
      The extension determines the way each of the arrays will be saved.
    """

    import os
    from ...io import save

    fobj = self.session.query(File).filter_by(id=id).one()
    fullpath = os.path.join(directory, str(fobj.path) + extension)
    fulldir = os.path.dirname(fullpath)
    utils.makedirs_safe(fulldir)
    save(obj, fullpath)

  def save(self, data, directory, extension):
    """This method takes a dictionary of blitz arrays or bob.database.Array's
    and saves the data respecting the original arrangement as returned by
    files().

    Keyword Parameters:

    data
      A dictionary with two keys 'real' and 'attack', each containing a
      dictionary mapping file ids from the original database to an object that
      supports the bob "save()" protocol.

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs()

    extension
      The extension determines the way each of the arrays will be saved.
    """    

    for key, value in data:
      self.save_one(key, value, directory, extension)
