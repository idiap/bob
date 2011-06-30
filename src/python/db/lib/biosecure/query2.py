#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
Biosecure database in the most obvious ways.
"""

import utils
from models import *

class Database(object):
  """The dataset class opens and maintains a connection opened to the Database.

  It provides many different ways to probe for the characteristics of the data
  and for the data itself inside the database.
  """

  def __init__(self):
    # opens a session to the database - keep it open until the end
    self.session = utils.session()


  def __check_validity__(self, l, obj, valid):
    """Checks validity of user input data against a set of valid values"""
    if not l: return valid
    elif isinstance(l, str): return self.__check_validity__((l,), obj, valid)
    for k in l:
      if k not in valid:
        raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
    return l


  def clients(self, protocol=None, groups=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ("Mc")

    groups
      The groups to which the clients belong ("dev", "eval", "world")

    Returns: A list containing all the client ids which have the given
    properties.
    """

    VALID_PROTOCOLS = ('ca0', 'caf', 'wc')
    VALID_GROUPS = ('dev', 'eval', 'world')
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    # List of the clients
    q = self.session.query(Client).filter(Client.sgroup.in_(groups)).\
          order_by(Client.id)
    retval = {'clients': [] }
    for id in [k.id for k in q]: 
      retval['clients'].append(id)
    return retval


  def files(self, directory=None, extension=None, protocol=None,
      purpose=None, client_id=None, groups=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the Biosecure protocols ("ca0", "caf", "wc").

    purpose
      The purpose required to be retrieved ("enrol", "probe") or a tuple
      with several of them. If 'None' is given (this is the default), it is 
      considered the same as a tuple with all possible values.

    client_id
      Only retrieves the files for the provided list of client ids. If 'None' is 
      given (this is the default), no filter over the client_ids is performed.

    groups
      One of the groups ("dev", "eval", "world") or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A list containing the resolved filenames considering all
    the filtering criteria or a full Torch Arrayset if you allow me to check
    for the existence of files (set check=True).
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('ca0', 'caf', 'wc')
    VALID_PURPOSE = ('enrol', 'probe', 'world')
    VALID_GROUPS = ('dev', 'eval', 'world')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purpose = self.__check_validity__(purpose, "purpose", VALID_PURPOSE)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    retval = {'world': {}, 'dev': {}, 'eval': {} }

    # List of the clients
    purpose_list = ('enrol', 'probe', 'world')
    q = 0
    
    if client_id:
      q = self.session.query(Client).filter(and_(Client.sgroup.in_(groups), Client.id.in_(client_id))).order_by(Client.id)
      for k in q:
        if k.sgroup == 'world':
          retval['world'][str(k.id)] = {}
        else:
          retval[k.sgroup][str(k.id)] = {'enrol': {}, 'probe': {} }
          retval[k.sgroup][str(k.id)]['enrol'] = {'client': {}, 'impostor': {} }
          retval[k.sgroup][str(k.id)]['probe'] = {'client': {}, 'impostor': {} }
    else: 
      client_id = []
      q = self.session.query(Client).filter(Client.sgroup.in_(groups)).order_by(Client.id)
      for k in q:
        client_id.append(k.id)
        if k.sgroup == 'world':
          retval['world'][str(k.id)] = {}
        else:
          retval[k.sgroup][str(k.id)] = {'enrol': {}, 'probe': {} }
          retval[k.sgroup][str(k.id)]['enrol'] = {'client': {}, 'impostor': {} }
          retval[k.sgroup][str(k.id)]['probe'] = {'client': {}, 'impostor': {} }

    # Add world data if required
    if 'world' in groups:
      q = self.session.query(File, Protocol, ProtocolPurpose).join(Client).\
            filter(and_(Client.id.in_(client_id), Client.sgroup == 'world')).\
            filter(Protocol.name == ProtocolPurpose.name).\
            filter(and_(Protocol.name.in_(protocol), ProtocolPurpose.sgroup == 'world', ProtocolPurpose.purpose.in_(purpose))).\
            filter(and_(File.session == ProtocolPurpose.session, File.camera == Protocol.camera)).\
            order_by(File.camera, File.client_id, File.session, File.shot)
      for k in q:
        if not str(k[0].client_id) in retval['world']:
          retval['world'][str(k[0].client_id)] = {}
        retval['world'][str(k[0].client_id)][str(k[0].id)] = make_path(k[0].path, directory, extension)
 
    # Add dev data if required
    if 'dev' in groups:
      q = self.session.query(File, Protocol, ProtocolPurpose).join(Client).\
            filter(Client.sgroup == 'dev').\
            filter(Protocol.name == ProtocolPurpose.name).\
            filter(and_(Protocol.name.in_(protocol), ProtocolPurpose.sgroup == 'dev', ProtocolPurpose.purpose.in_(purpose))).\
            filter(and_(File.session == ProtocolPurpose.session, File.camera == Protocol.camera)).\
            order_by(File.camera, File.client_id, File.session, File.shot)
      for k in q:
        if k[2].purpose == 'probe':
          for client in retval['dev']:
            if client != str(k[0].client_id):
              retval['dev'][client]['probe']['impostor'][str(k[0].id)] = make_path(k[0].path, directory, extension)
            else:
              if str(k[0].client_id) in retval['dev']:
                retval['dev'][str(k[0].client_id)]['probe']['client'][str(k[0].id)] = make_path(k[0].path, directory, extension)
        elif k[2].purpose == 'enrol':
          if str(k[0].client_id) in retval['dev']:
            retval['dev'][str(k[0].client_id)]['enrol']['client'][str(k[0].id)] = make_path(k[0].path, directory, extension)

    # Add eval data if required
    if 'eval' in groups:
      q = self.session.query(File, Protocol, ProtocolPurpose).join(Client).\
            filter(Client.sgroup == 'eval').\
            filter(Protocol.name == ProtocolPurpose.name).\
            filter(and_(Protocol.name.in_(protocol), ProtocolPurpose.sgroup == 'eval', ProtocolPurpose.purpose.in_(purpose))).\
            filter(and_(File.session == ProtocolPurpose.session, File.camera == Protocol.camera)).\
            order_by(File.camera, File.client_id, File.session, File.shot)
      for k in q:
        if k[2].purpose == 'probe':
          for client in retval['eval']:
            if client != str(k[0].client_id):
              retval['eval'][client]['probe']['impostor'][str(k[0].id)] = make_path(k[0].path, directory, extension)
            else:
              if str(k[0].client_id) in retval['eval']:
                retval['eval'][str(k[0].client_id)]['probe']['client'][str(k[0].id)] = make_path(k[0].path, directory, extension)
        elif k[2].purpose == 'enrol':
          if str(k[0].client_id) in retval['eval']:
            retval['eval'][str(k[0].client_id)]['enrol']['client'][str(k[0].id)] = make_path(k[0].path, directory, extension)

    return retval


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

    for c_id in data.keys():
      if c_id == 'world':
        for key, value in data[c_id]: 
          self.save_one(key, value, directory, extension)
      else:
        for key, value in data[c_id]['enrol']: 
          self.save_one(key, value, directory, extension)
        for key, value in data[c_id]['probe']: 
          self.save_one(key, value, directory, extension)
