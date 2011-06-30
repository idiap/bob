#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
BANCA database in the most obvious ways.
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

  def clients(self, groups=None, gender=None, language=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    groups
      The groups to which the clients belong ("g1", "g2", "world")

    gender
      The genders to which the clients belong ("f", "m")

    language
      TODO: only English is currently supported
      The language spoken by the clients ("en",)

    Returns: A list containing all the client ids which have the given
    properties.
    """

    VALID_GROUPS = ('g1', 'g2', 'world')
    VALID_GENDERS = ('m', 'f')
    VALID_LANGUAGES = ('en',)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    gender = self.__check_validity__(gender, "gender", VALID_GENDERS)
    language = self.__check_validity__(language, "language", VALID_LANGUAGES)
    # List of the clients
    q = self.session.query(Client).filter(Client.sgroup.in_(groups)).\
          filter(Client.gender.in_(gender)).\
          filter(Client.language.in_(language)).\
          order_by(Client.id)
    retval = []
    for id in [k.id for k in q]: 
      retval.append(id)
    return retval

  def models(self, protocol=None, groups=None):
    """Returns a set of models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").
    
    groups
      The groups to which the subjects attached to the models belong ("g1", "g2", "world")

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.clients(groups)

  def files(self, directory=None, extension=None, protocol=None,
      purposes=None, client_ids=None, groups=None, classes=None, 
      languages=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").

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
      One of the groups ("g1", "g2", "world") or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the BANCA database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    def check_validity(l, obj, valid):
      """Checks validity of user input data against a set of valid values"""
      if not l: return valid
      elif isinstance(l, str): return check_validity((l,), obj, valid)
      for k in l:
        if k not in valid:
          raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
      return l

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('Mc', 'Md', 'Ma', 'Ud', 'Ua', 'P', 'G')
    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('g1', 'g2', 'world')
    VALID_LANGUAGES = ('en', 'fr', 'sp')
    VALID_CLASSES = ('client', 'impostor')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    languages = self.__check_validity__(languages, "language", VALID_LANGUAGES)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)
    retval = {}
    
    if 'world' in groups:
      if not client_ids:
        q = self.session.query(File).join(Client).\
              filter(Client.sgroup == 'world').\
              filter(Client.language.in_(languages)).\
              order_by(File.real_id, File.session_id, File.claimed_id, File.shot)
      else:
        q = self.session.query(File).join(Client).\
              filter(File.claimed_id.in_(client_ids)).\
              filter(Client.sgroup == 'world').\
              filter(Client.language.in_(languages)).\
              order_by(File.real_id, File.session_id, File.claimed_id, File.shot) 
      for k in q:
        retval[k.id] = make_path(k.path, directory, extension)
    
    if ('g1' in groups or 'g2' in groups):
      if('enrol' in purposes):
        if not client_ids:
          q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                filter(File.claimed_id == File.real_id).\
                filter(Client.sgroup.in_(groups)).\
                filter(Client.language.in_(languages)).\
                filter(Protocol.name.in_(protocol)).\
                filter(Protocol.purpose == 'enrol').\
                order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
        else:
          q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                filter(File.claimed_id == File.real_id).\
                filter(File.claimed_id.in_(client_ids)).\
                filter(Client.sgroup.in_(groups)).\
                filter(Client.language.in_(languages)).\
                filter(Protocol.name.in_(protocol)).\
                filter(Protocol.purpose == 'enrol').\
                order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
        for k in q:
          retval[k.id] = make_path(k.path, directory, extension)
      if('probe' in purposes):
        if('client' in classes):
          if not client_ids:
            q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                  filter(File.claimed_id == File.real_id).\
                  filter(Client.sgroup.in_(groups)).\
                  filter(Client.language.in_(languages)).\
                  filter(Protocol.name.in_(protocol)).\
                  filter(Protocol.purpose == 'probe').\
                  order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
          else:
            q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                  filter(File.claimed_id.in_(client_ids)).\
                  filter(File.claimed_id == File.real_id).\
                  filter(Client.sgroup.in_(groups)).\
                  filter(Client.language.in_(languages)).\
                  filter(Protocol.name.in_(protocol)).\
                  filter(Protocol.purpose == 'probe').\
                  order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
          for k in q:
            retval[k.id] = make_path(k.path, directory, extension)
        if('impostor' in classes):
          if not client_ids:
            q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                  filter(File.claimed_id != File.real_id).\
                  filter(Client.sgroup.in_(groups)).\
                  filter(Client.language.in_(languages)).\
                  filter(Protocol.name.in_(protocol)).\
                  filter(or_(Protocol.purpose == 'probe', Protocol.purpose == 'probeImpostor')).\
                  order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
          else:
            q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                  filter(File.claimed_id.in_(client_ids)).\
                  filter(File.claimed_id != File.real_id).\
                  filter(Client.sgroup.in_(groups)).\
                  filter(Client.language.in_(languages)).\
                  filter(Protocol.name.in_(protocol)).\
                  filter(or_(Protocol.purpose == 'probe', Protocol.purpose == 'probeImpostor')).\
                  order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
          for k in q:
            retval[k.id] = make_path(k.path, directory, extension)
        
    return retval

  # TODO: dictionary interface
  #def dictionary(self, directory=None, extension=None, protocol=None,
  #    purpose=None, client_id=None, groups=None):
    """Returns a multiple-layers dictionary for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").

    purpose
      The purpose required to be retrieved ("enrol", "probe", "world") or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values.

    client_ids
      Only retrieves the files for the provided list of client ids. If 'None' is 
      given (this is the default), no filter over the client_ids is performed.

    groups
      One of the two groups ("g1", "g2") or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A multiple-layers dictionary containing the resolved filenames 
    considering all the filtering criteria or a full Torch Arrayset if you 
    allow me to check for the existence of files (set check=True).
      * The first key belongs to ('world', 'g1', 'g2'), 
      * The second key is the claimed client id
      * The third key corresponds to the type of access ('client', 'impostor')
      * The fourth key is the key id of the file
    """

    """
    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('Mc', 'Md', 'Ma', 'Ud', 'Ua', 'P', 'G')
    VALID_PURPOSES = ('enrol', 'probe', 'world')
    # TODO: check client id, factorize valid groups ?
    VALID_GROUPS = ('g1', 'g2')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purpose = self.__check_validity__(purpose, "purpose", VALID_PURPOSE)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    retval = {'world': {}, 'enrol': {}, 'probe': {} }

    # Add world data if required
    if 'world' in purpose:
      q = self.session.query(File).join(Client).filter(File.real_id == Client.id).filter(Client.sgroup == 'world').order_by(File.real_id, File.session_id, File.claimed_id, File.shot)
      for k in q: 
        if not str(k.claimed_id) in retval['world']:
          retval['world'][str(k.claimed_id)] = {'client': {}, 'impostor': {} }
        if k.real_id == k.claimed_id:
          retval['world'][str(k.claimed_id)]['client'][str(k.id)] = make_path(k.path, directory, extension)
        else:
          retval['world'][str(k.claimed_id)]['impostor'][str(k.id)] = make_path(k.path, directory, extension)

    # List of the clients
    if not client_ids:
      client_ids = []
      q = self.session.query(File).join(Client).filter(File.real_id == Client.id).filter(Client.sgroup.in_(groups)).order_by(Client.id)
      for k in q:
        if k.real_id not in client_ids:
          client_ids.append(k.claimed_id)
          retval['enrol'][str(k.claimed_id)] = {'client': {}, 'impostor': {} }
          retval['probe'][str(k.claimed_id)] = {'client': {}, 'impostor': {} }
    else:
      for id in client_ids:
        retval['enrol'][str(id)] = {'client': {}, 'impostor': {} }
        retval['probe'][str(id)] = {'client': {}, 'impostor': {} }
    
    # Loop over the client_ids and add required data
    if 'enrol' in purpose:
      q = self.session.query(File).join(Client).join(Session).join(Protocol).filter(File.claimed_id.in_(client_ids)).filter(File.real_id == File.claimed_id).filter(Client.sgroup.in_(groups)).filter(File.session_id == Session.id).filter(Protocol.name.in_(protocol)).filter(Protocol.session_id == Session.id).filter(Protocol.purpose == 'enrol').order_by(File.real_id, File.session_id, File.claimed_id, File.shot)
      for k in q:
        retval['enrol'][str(k.claimed_id)]['client'][str(k.id)] = make_path(k.path, directory, extension)
    
    if 'probe' in purpose:
      q = self.session.query(File).join(Client).join(Session).join(Protocol).filter(File.claimed_id.in_(client_ids)).filter(Client.sgroup.in_(groups)).filter(Protocol.session_id == File.session_id).filter(Protocol.name.in_(protocol)).filter(or_(Protocol.purpose == 'probe', and_(Protocol.purpose == 'probeImpostor', File.claimed_id != File.real_id))).order_by(File.real_id, File.session_id, File.claimed_id, File.shot)
      for k in q:
        if k.real_id == k.claimed_id:
          retval['probe'][str(k.claimed_id)]['client'][str(k.id)] = make_path(k.path, directory, extension)
        else:
          retval['probe'][str(k.claimed_id)]['impostor'][str(k.id)] = make_path(k.path, directory, extension)

    return retval
    """

  def save_one(self, id, obj, directory, extension):
    """Saves a single object supporting the torch save() protocol.

    This method will call save() on the the given object using the correct
    database filename stem for the given id.
    
    Keyword Parameters:

    id
      The id of the object in the database table "file".

    obj
      The object that needs to be saved, respecting the torch save() protocol.

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs()

    extension
      The extension determines the way each of the arrays will be saved.
    """

    import os
    fobj = self.session.query(File).filter_by(id=id).one()
    fullpath = os.path.join(directory, str(fobj.path) + extension)
    fulldir = os.path.dirname(fullpath)
    if not os.path.exists(fulldir): os.makedirs(fulldir)
    obj.save(fullpath)

  def save(self, data, directory, extension):
    """This method takes a dictionary of blitz arrays or torch.database.Array's
    and saves the data respecting the original arrangement as returned by
    files().

    Keyword Parameters:

    data
      A dictionary with two keys 'real' and 'attack', each containing a
      dictionary mapping file ids from the original database to an object that
      supports the Torch "save()" protocol.

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs()

    extension
      The extension determines the way each of the arrays will be saved.
    """    

    for key, value in data:
      self.save_one(key, value, directory, extension)
