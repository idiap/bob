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
      The protocol to consider ("Mc")

    groups
      The groups to which the clients belong ("dev", "eval", "world")

    gender
      The genders to which the clients belong ("f", "m")

    birthyear
      The birth year of the clients (in the range [1900,2050])

    Returns: A list containing all the client ids which have the given
    properties.
    """

    VALID_PROTOCOLS = ('Mc')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_GENDERS = ('m', 'f')
    VALID_BIRTHYEARS =  range(1900, 2050)
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    gender = self.__check_validity__(gender, "gender", VALID_GENDERS)
    birthyear = self.__check_validity__(birthyear, "birthyear", VALID_BIRTHYEARS)
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

  def models(self, protocol=None, groups=None):
    """Returns a set of models for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ("Mc")
    
    groups
      The groups to which the subjects attached to the models belong ("dev", "eval", "world")

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.clients(protocol, groups)

  def files(self, directory=None, extension=None, protocol=None,
      purposes=None, model_ids=None, groups=None, classes=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("Mc").

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

    VALID_PROTOCOLS = ('Mc', )
    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_CLASSES = ('client', 'impostor')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)
    retval = {}
    
    if 'world' in groups:
      # Highres
      """
      if not model_ids:
        q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
              filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup == 'world', \
                          ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
              join(Client).\
              filter(and_(Client.id == ProtocolClientGroup.client_id, ProtocolClientGroup.sgroup == 'world',\
                          ProtocolClientGroup.name.in_(protocol), FileHighres.client_id == ProtocolClientGroup.client_id)).\
              order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
      else:
        q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
              filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup == 'world', \
                          ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
              join(Client).\
              filter(and_(Client.id.in_(model_ids), Client.id == ProtocolClientGroup.client_id, ProtocolClientGroup.sgroup == 'world',\
                          ProtocolClientGroup.name.in_(protocol), FileHighres.client_id == ProtocolClientGroup.client_id)).\
              order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
      for k in q:
        retval[k[0].id] = make_path(k[0].path, directory, extension)
      """
      # Multiview
      if not model_ids:
        q = self.session.query(File, ProtocolMultiview).join(Client).join(FileMultiview).\
              filter(and_(ProtocolMultiview.name.in_(protocol), ProtocolMultiview.sgroup == 'world', \
                          ProtocolMultiview.session_id == File.session_id, ProtocolMultiview.shot_id == FileMultiview.shot_id, \
                          ProtocolMultiview.recording_id == File.recording_id, ProtocolMultiview.camera_id == FileMultiview.camera_id)).\
              filter(Client.sgroup == 'world').\
              order_by(File.client_id, File.session_id, FileMultiview.shot_id)
      else:
        """
        q = self.session.query(File, ProtocolMultiview).join(Client).join(FileMultiview).\
              filter(and_(ProtocolMultiview.name.in_(protocol), ProtocolMultiview.sgroup == 'world', \
                          ProtocolMultiview.session_id == File.session_id, ProtocolMultiview.shot_id == FileMultiview.shot_id, \
                          ProtocolMultiview.recording_id == File.recording_id, ProtocolMultiview.camera_id == FileMultiview.camera_id)).\
              filter(and_(Client.id.in_(model_ids), Client.sgroup == 'world')).\
              order_by(File.client_id, File.session_id, FileMultiview.shot_id)
        """
        q = self.session.query(File, ProtocolMultiview).join(Client).join(FileMultiview).\
              filter(and_(Client.id.in_(model_ids), Client.sgroup == 'world')).\
              order_by(File.client_id, File.session_id, FileMultiview.shot_id)
      print q.count()
      for k in q:
        retval[k[0].id] = make_path(k[0].path, directory, extension)
    
    if('dev' in groups or 'eval' in groups):
      dev_eval_groups = groups
      # Highres
      """
      if('enrol' in purposes):
        # TODO: check ids for enrolling
        if not model_ids:
          q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
                filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup.in_(dev_eval_groups), \
                            ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
                join(Client).\
                filter(and_(Client.id == ProtocolClientGroup.client_id,\
                            ProtocolClientGroup.sgroup.in_(dev_eval_groups), ProtocolClientGroup.name.in_(protocol),\
                            FileHighres.client_id == ProtocolClientGroup.client_id)).\
                order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
        else:
          q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
                filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup.in_(dev_eval_groups), \
                            ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
                join(Client).\
                filter(and_(Client.id.in_(model_ids), Client.id == ProtocolClientGroup.client_id,\
                            ProtocolClientGroup.sgroup.in_(dev_eval_groups), ProtocolClientGroup.name.in_(protocol),\
                            FileHighres.client_id == ProtocolClientGroup.client_id)).\
                order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
        for k in q:
          retval[k[0].id] = make_path(k[0].path, directory, extension)
      if('probe' in purposes):
        # TODO: check ids for enrolling
        if('client' in classes):
          if not model_ids:
            q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
                  filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup.in_(dev_eval_groups), \
                              ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
                  join(Client).\
                  filter(and_(Client.id == ProtocolClientGroup.client_id,\
                              ProtocolClientGroup.sgroup.in_(dev_eval_groups), ProtocolClientGroup.name.in_(protocol),\
                              FileHighres.client_id == ProtocolClientGroup.client_id)).\
                  order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
          else:
            q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
                  filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup.in_(dev_eval_groups), \
                              ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
                  join(Client).\
                  filter(and_(Client.id.in_(model_ids), Client.id == ProtocolClientGroup.client_id,\
                              ProtocolClientGroup.sgroup.in_(dev_eval_groups), ProtocolClientGroup.name.in_(protocol),\
                              FileHighres.client_id == ProtocolClientGroup.client_id)).\
                  order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
          for k in q:
            retval[k[0].id] = make_path(k[0].path, directory, extension)
        if('impostor' in classes):
          if not model_ids:
            q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
                  filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup.in_(dev_eval_groups), \
                              ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
                  join(Client).\
                  filter(and_(Client.id == ProtocolClientGroup.client_id,\
                              ProtocolClientGroup.sgroup.in_(dev_eval_groups), ProtocolClientGroup.name.in_(protocol),\
                              FileHighres.client_id == ProtocolClientGroup.client_id)).\
                  order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
          else:
            q = self.session.query(FileHighres, ProtocolHighres, ProtocolClientGroup).\
                  filter(and_(ProtocolHighres.name.in_(protocol), ProtocolHighres.sgroup.in_(dev_eval_groups), \
                              ProtocolHighres.session_id == FileHighres.session_id, ProtocolHighres.shot == FileHighres.shot)).\
                  join(Client).\
                  filter(and_(Client.id.in_(model_ids), Client.id == ProtocolClientGroup.client_id,\
                              ProtocolClientGroup.sgroup.in_(dev_eval_groups), ProtocolClientGroup.name.in_(protocol),\
                              FileHighres.client_id == ProtocolClientGroup.client_id)).\
                  order_by(FileHighres.client_id, FileHighres.session_id, FileHighres.shot)
          for k in q:
            retval[k[0].id] = make_path(k[0].path, directory, extension)
      """
      # TODO: Multiview
 
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

    for key, value in data:
      self.save_one(key, value, directory, extension)
