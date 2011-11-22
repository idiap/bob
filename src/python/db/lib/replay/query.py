#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 17 May 13:58:09 2011 

"""This module provides the Dataset interface allowing the user to query the
replay attack database in the most obvious ways.
"""

import os
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

  def files(self, directory=None, extension=None, support=('fixed', 'hand'),
      protocol='grandtest', groups=('train', 'devel', 'test'), 
      cls=('attack', 'real')):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    support
      One of the valid support types ("fixed" or "hand") or both, as a tuple.
      If you set this parameter to an empty string or the value None, we use 
      reset it to the default.

    protocol
      The protocol for the attack ("mobile", "highdef", "print", "photo",
      "video" or "grandtest").  If you set this parameter to an empty string or
      the value None, we use reset it to the default.

    groups
      One of the protocolar subgroups of data ("train", "devel", "test") or a
      tuple with several of them.  If you set this parameter to an empty string
      or the value None, we use reset it to the default.

    cls
      Either "attack", "real", "enroll" or any combination of those (in a
      tuple). Defines the class of data to be retrieved.  If you set this
      parameter to an empty string or the value None, we use reset it to the
      default.

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the replay attack database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    def check_validity(l, obj, valid, default):
      """Checks validity of user input data against a set of valid values"""
      if not l: return default
      elif isinstance(l, str): return check_validity((l,), obj, valid, default)
      for k in l:
        if k not in valid:
          raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
      return l

    def make_path(stem, directory, extension):
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_GROUPS = ('train', 'devel', 'test')
    VALID_SUPPORTS = ('fixed', 'hand')
    VALID_PROTOCOLS = ('mobile', 'highdef', 'print', 'photo', 'video', 
        'grandtest')
    VALID_CLASSES = ('real', 'attack', 'enroll')

    groups = check_validity(groups, "group", VALID_GROUPS, VALID_GROUPS)
    support = check_validity(support, "support", VALID_SUPPORTS, VALID_SUPPORTS)
    cls = check_validity(cls, "class", VALID_CLASSES, ('real', 'attack'))

    if not protocol: protocol = 'grandtest' #default
    if protocol not in VALID_PROTOCOLS:
      raise RuntimeError, 'Invalid protocol "%s". Valid values are %s' % \
          (protocol, VALID_PROTOCOLS)

    retval = {}

    # real-accesses are simpler to query
    if 'enroll' in cls:
      q = self.session.query(RealAccess).join(File).join(Client).filter(Client.set.in_(groups)).filter(RealAccess.purpose=='enroll').order_by(Client.id)
      for key, value in [(k.file.id, k.file.path) for k in q]: 
        retval[key] = make_path(str(value), directory, extension)
      
    # real-accesses are simpler to query
    if 'real' in cls:
      q = self.session.query(RealAccess).join(File).join(Client).filter(Client.set.in_(groups)).filter(RealAccess.purpose=='authenticate').order_by(Client.id)
      for key, value in [(k.file.id, k.file.path) for k in q]: 
        retval[key] = make_path(str(value), directory, extension)
      
    # attacks will have to be filtered a little bit more
    if 'attack' in cls:
      q = self.session.query(Attack).join(File).join(Client).\
          filter(Client.set.in_(groups)).\
          filter(Attack.attack_support.in_(support))

      if protocol in ('mobile', 'highdef', 'print'):
        # these are protocols that filter by attack_device
        q = q.filter(Attack.attack_device==protocol)

      elif protocol in ('photo', 'video'):
        # these are protocols that filter by sample_type
        q = q.filter(Attack.sample_type==protocol)

      q = q.order_by(Client.id)

      for key, value in [(k.file.id, k.file.path) for k in q]: 
        retval[key] = make_path(str(value), directory, extension)

    return retval

  def paths(self, ids, prefix='', suffix=''):
    """Returns a full file paths considering particular file ids, a given
    directory and an extension
    
    Keyword Parameters:

    id
      The ids of the object in the database table "file". This object should be
      a python iterable (such as a tuple or list).

    prefix
      The bit of path to be prepended to the filename stem

    suffix
      The extension determines the suffix that will be appended to the filename
      stem.

    Returns a list (that may be empty) of the fully constructed paths given the
    file ids.
    """
    fobj = self.session.query(File).filter(File.id.in_(ids))
    retval = []
    for p in ids:
      retval.extend([os.path.join(prefix, str(k.path) + suffix) 
        for k in fobj if k.id == p])
    return retval

  def reverse(self, paths):
    """Reverses the lookup: from certain stems, returning file ids
    
    Keyword Parameters:

    paths
      The filename stems I'll query for. This object should be a python
      iterable (such as a tuple or list)

    Returns a list (that may be empty).
    """

    fobj = self.session.query(File).filter(File.path.in_(paths))
    retval = []
    for p in paths:
      retval.extend([k.id for k in fobj if k.path == p])
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

    from ...io import save

    fobj = self.session.query(File).filter_by(id=id).one()
    fullpath = os.path.join(directory, str(fobj.path) + extension)
    fulldir = os.path.dirname(fullpath)
    utils.makedirs_safe(fulldir)
    save(obj, fullpath)

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
