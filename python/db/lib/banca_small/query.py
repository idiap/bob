#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
BANCA_SMALL database in the most obvious ways.
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

  def __group_replace_alias__(self, l):
    """Replace 'dev' by 'g1' in a list of groups, and 
       returns the new list"""
    if not l: return l
    elif isinstance(l, str): return self.__group_replace_alias__((l,))
    l2 = []
    for val in l:
      if(val == 'dev'): l2.append('g1')
      else: l2.append(val)
    return tuple(l2)

  def __check_validity__(self, l, obj, valid):
    """Checks validity of user input data against a set of valid values"""
    if not l: return valid
    elif isinstance(l, str): return self.__check_validity__((l,), obj, valid)
    for k in l:
      if k not in valid:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
    return l

  def clients(self, protocol=None, groups=None, gender=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA_SMALL protocols ("P")

    groups
      The groups to which the clients belong ("g1", "world").
      Note that 'dev' is an alias to 'g1'

    gender
      The genders to which the clients belong ("f", "m")

    Returns: A list containing all the client ids which have the given
    properties.
    """

    groups = self.__group_replace_alias__(groups)
    VALID_GROUPS = ('g1', 'world')
    VALID_GENDERS = ('m', 'f')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    gender = self.__check_validity__(gender, "gender", VALID_GENDERS)

    retval = []
    # List of the clients
    if "world" in groups:
      q = self.session.query(Client).filter(Client.sgroup == 'world').\
            filter(Client.gender.in_(gender)).\
          order_by(Client.id)
      for id in [k.id for k in q]: 
        retval.append(id)

    if 'g1' in groups:
      q = self.session.query(Client).filter(Client.sgroup == 'g1').\
            filter(Client.id != '1002').\
            filter(Client.gender.in_(gender)).\
            order_by(Client.id)
      for id in [k.id for k in q]: 
        retval.append(id)

    return retval

  def tnorm_ids(self, protocol=None):
    """Returns a set of T-Norm ids for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA_SMALL protocols ("P")

    Returns: A list containing all the client ids which have the given
    properties.
    """

    retval = []
    # List of the clients
    q = self.session.query(TNormClient).\
          order_by(TNormClient.id)
    for id in [k.id for k in q]: 
      retval.append(id)

    return retval

  def get_real_id_from_tnorm_id(self, tnorm_id, protocol=None):
    """Returns the real id of a client from a TNorm client id.

    Keyword Parameters:

    tnorm_client_id
      the id of the T-Norm client

    protocol
      One of the BANCA_SMALL protocols ("P")

    Returns: The real_id of the client associated with the T-Norm id    
    """

    # List of the clients
    q = self.session.query(TNormClient).filter(TNormClient.id == tnorm_id).\
          order_by(TNormClient.id)
    if q.count() !=1:
      #throw exception?
      return None
    else:
      return q.first().real_id

  def models(self, protocol=None, groups=None, ids_real=None):
    """Returns a set of models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA_SMALL protocols ("P").
    
    groups
      The groups to which the subjects attached to the models belong ("g1", "world")
      Note that 'dev' is an alias to 'g1'

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.clients(protocol, groups, ids_real)

  def get_client_id_from_model_id(self, model_id):
    """Returns the client_id attached to the given model_id
    
    Keyword Parameters:

    model_id
      The model_id to consider

    Returns: The client_id attached to the given model_id
    """
    return model_id

  def get_client_id_from_file_id(self, file_id):
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
      return q.first().real_id

  def get_internal_path_from_file_id(self, file_id):
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
      purposes=None, model_ids=None, real_ids=None, groups=None, classes=None): 
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA_SMALL protocols ("P").

    purposes
      The purposes required to be retrieved ("enrol", "probe") or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values. This field is
      ignored for the data from the "world" group.

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    real_ids
      Only retrieves the files for the provided list of real ids (real client
      id). If 'None' is given (this is the default), no filter over real_ids
      is performed.

    groups
      One of the groups ("g1", "world") or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.
      Note that 'dev' is an alias to 'g1'

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)

    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA_SMALL database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    groups = self.__group_replace_alias__(groups)
    VALID_PROTOCOLS = ('P')
    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('g1', 'world')
    VALID_CLASSES = ('client', 'impostor')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)

    retval = {}

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
    
    if 'world' in groups:
      q = self.session.query(File).join(Client).\
            filter(Client.sgroup == 'world')
      if model_ids:
        q = q.filter(File.claimed_id.in_(model_ids))
      if real_ids:
        q = q.filter(File.real_id.in_(real_ids))
      q = q.order_by(File.real_id, File.session_id, File.claimed_id, File.shot) 
      for k in q:
        retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)
    
    if ('g1' in groups):
      if('enrol' in purposes):
        q = self.session.query(File,Protocol).join(Client).\
              filter(File.claimed_id == File.real_id).\
              filter(File.session_id == Protocol.session_id).\
              filter(Client.sgroup.in_(groups)).\
              filter(Protocol.name.in_(protocol)).\
              filter(Protocol.purpose == 'enrol')
        if model_ids:
          q = q.filter(File.claimed_id.in_(model_ids))
        if real_ids:
          q = q.filter(File.real_id.in_(real_ids))
        q = q.order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
        for k in q:
          k = k[0]
          retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)
      if('probe' in purposes):
        if('client' in classes):
          q = self.session.query(File,Protocol).join(Client).\
                filter(File.claimed_id == File.real_id).\
                filter(File.session_id == Protocol.session_id).\
                filter(Client.sgroup.in_(groups)).\
                filter(Protocol.name.in_(protocol)).\
                filter(Protocol.purpose == 'probe')
          if model_ids:
            q = q.filter(File.claimed_id.in_(model_ids))
          if real_ids:
            q = q.filter(File.real_id.in_(real_ids))
          q = q.order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
          for k in q:
            k = k[0]
            retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)
        if('impostor' in classes):
          q = self.session.query(File,Protocol).join(Client).\
                filter(File.claimed_id != File.real_id).\
                filter(File.session_id == Protocol.session_id).\
                filter(Client.sgroup.in_(groups)).\
                filter(Protocol.name.in_(protocol)).\
                filter(or_(Protocol.purpose == 'probe', Protocol.purpose == 'probeImpostor'))
          if model_ids:
            q = q.filter(File.claimed_id.in_(model_ids))
          if real_ids:
            q = q.filter(File.real_id.in_(real_ids))
          for k in q:
            k = k[0]
            retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)

    return retval

  def files(self, directory=None, extension=None, protocol=None,
      purposes=None, model_ids=None, real_ids=None, groups=None, classes=None): 
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA_SMALL protocols ("P").

    purposes
      The purposes required to be retrieved ("enrol", "probe") or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values. This field is
      ignored for the data from the "world" group.

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    real_ids
      Only retrieves the files for the provided list of real ids (real client
      id). If 'None' is given (this is the default), no filter over real_ids
      is performed.

    groups
      One of the groups ("g1", "world") or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.
      Note that 'dev' is an alias to 'g1'

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the BANCA_SMALL database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    retval = {}
    d = self.objects(directory, extension, protocol, purposes, model_ids, real_ids, groups, classes)
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
