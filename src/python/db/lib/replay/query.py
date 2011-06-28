#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 17 May 13:58:09 2011 

"""This module provides the Dataset interface allowing the user to query the
replay attack database in the most obvious ways.
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

  def files(self, directory=None, extension=None, support=None,
      device=None, groups=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    support
      One of the valid support types ("fixed" or "hand"). If 'None' is given
      (this is the default), it is considered the same as a tuple with all
      possible values.

    device
      The device type used for the attack ("mobile", "highdef", "print"). If
      'None' is given (this is the default), it is considered the same as a
      tuple with all possible values.

    groups
      One of the protocolar subgroups of data ("train", "devel", "test") or a
      tuple with several of them. If 'None' is given (this is the default), it
      is considered the same as a tuple with all possible values.

    Returns: A list containing the resolved filenames considering all
    the filtering criteria or a full Torch Arrayset if you allow me to check
    for the existence of files (set check=True).
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

    VALID_GROUPS = ('train', 'devel', 'test')
    VALID_SUPPORTS = ('fixed', 'hand')
    VALID_DEVICES = ('mobile', 'highdef', 'print')

    groups = check_validity(groups, "group", VALID_GROUPS)
    device = check_validity(device, "device", VALID_DEVICES)
    support = check_validity(support, "support", VALID_SUPPORTS)

    retval = {'real': {}, 'attack': {}}

    # real-accesses are simpler to query
    q = self.session.query(RealAccess).join(File).join(Client).filter(
        Client.set.in_(groups)).order_by(Client.id)
    for key, value in [(k.file.id, k.file.path) for k in q]: 
      retval['real'][key] = make_path(str(value), directory, extension)
      
    # attacks will have to be filtered a little bit more
    q = self.session.query(Attack).join(File).join(Client).\
        filter(Client.set.in_(groups)).\
        filter(Attack.attack_device.in_(device)).\
        filter(Attack.attack_support.in_(support)).order_by(Client.id)
    for key, value in [(k.file.id, k.file.path) for k in q]: 
      retval['real'][key] = make_path(str(value), directory, extension)

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
    
    for group in data.keys():
      for key, value in data[group]: 
        self.save_one(key, value, directory, extension)
