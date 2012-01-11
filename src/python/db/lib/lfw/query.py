#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
LFW database in the most obvious ways.
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

  def clients(self, protocol=None, groups=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('',)

    groups
      The groups to which the clients belong ('',)

    Returns: A list containing all the tuples (client_id, client_name) which 
    have the given properties.
    """

    # List of the clients
    q = self.session.query(Client)
    q = q.order_by(Client.id)
    retval = []
    for k in q: 
      retval.append((k.id, k.name))
    return retval

  def clientsIds(self, protocol=None, groups=None):
    """Returns a set of client ids for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('',)

    groups
      The groups to which the clients belong ('',)

    Returns: A list containing all the client ids which have the given
    properties.
    """

    # List of the clients
    retval = []
    q = self.session.query(Client)
    q = q.order_by(Client.id)
    retval = []
    for k in q: 
      retval.append(k.id)
    return retval

  def clientsNames(self, protocol=None, groups=None):
    """Returns a set of client names for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('',)

    groups
      The groups to which the clients belong ('',)

    Returns: A list containing all the client names which have the given
    properties.
    """

    # List of the clients
    retval = []
    q = self.session.query(Client)
    q = q.order_by(Client.id)
    retval = []
    for k in q: 
      retval.append(k.name)
    return retval

  def getClientNameFromClientId(self, client_id):
    """Returns the client name attached to the given client_id

    Keyword Parameters:

    client_id
      The client_id to consider

    Returns: The client name attached to the given client_id
    """
    q = self.session.query(Client).\
          filter(Client.id == client_id)
    if q.count() !=1:
      #throw exception?
      return None
    else:
      return q.first().name

  def getClientIdFromClientName(self, client_name):
    """Returns the client_id attached to the given client_name

    Keyword Parameters:

    client_name
      The client_name to consider

    Returns: The client id attached to the given client_name
    """
    q = self.session.query(Client).\
          filter(Client.name == client_name)
    if q.count() !=1:
      #throw exception?
      return None
    else:
      return q.first().id

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
      client_ids=None, groups=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      The protocol to consider ('',)

    client_ids
      Only retrieves the files for the provided list of client ids.  
      If 'None' is given (this is the default), no filter over 
      the client_ids is performed.

    groups
      The groups to which the clients belong ('',)

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the client id
      - 2: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the LFW database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('',)
    VALID_GROUPS = ('',)

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    retval = {}
    
    if(isinstance(client_ids,str)):
      client_ids = (client_ids,)

    q = self.session.query(File).join(Client)
    if client_ids:
      q = q.filter(Client.id.in_(client_ids))
    q = q.order_by(File.client_id, File.shot_id)
    for k in q:
      retval[k.id] = (make_path(k.path, directory, extension), k.client_id, k.path)
        
    return retval

  def files(self, directory=None, extension=None, protocol=None,
      client_ids=None, groups=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      The protocol to consider ('',)

    client_ids
      Only retrieves the files for the provided list of client ids.  
      If 'None' is given (this is the default), no filter over 
      the client_ids is performed.

    groups
      The groups to which the clients belong ('',)

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the client id
      - 2: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the LFW database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.objects(directory, extension, protocol, client_ids, groups)
    for k in d: retval[k] = d[k][0]

    return retval

  def pairs(self, directory=None, extension=None, protocol=None, view=None,
      subset=None, classes=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      The protocol to consider ('',)

    view
      The view to consider ('view1','view2')

    subset
      The subset to consider ('train','test', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10')

    classes
      The groups to which the clients belong ('matched', 'unmatched')

    Returns: A dictionary containing:
      - 0: the resolved filename 1
      - 1: the resolved filename 2
      - 2: client_id associated with filename 1
      - 3: client_id associated with filename 2
      - 4: file_id associated with filename 1
      - 5: file_id associated with filename 2
    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the LFW database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('',)
    VALID_VIEWS = ('view1', 'view2')
    VALID_SUBSETS = ('train', 'test', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10')
    VALID_CLASSES = ('matched', 'unmatched')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    view = self.__check_validity__(view, "view", VALID_VIEWS)
    subset = self.__check_validity__(subset, "subset", VALID_SUBSETS)
    classes = self.__check_validity__(classes, "classes", VALID_CLASSES)
    retval = {}
    
    # Create alias in order to be able to join the same table twice
    from sqlalchemy.orm import aliased
    File1 = aliased(File)
    File2 = aliased(File)
    q = self.session.query(Pair,File1,File2).\
          filter(Pair.file_id1 == File1.id).\
          filter(Pair.file_id2 == File2.id).\
          filter(and_(Pair.view.in_(view), Pair.subset.in_(subset)))
    if not 'matched' in classes:
      q = q.filter(Pair.client_id1 != Pair.client_id2)
    if not 'unmatched' in classes:
      q = q.filter(Pair.client_id1 == Pair.client_id2)
    q = q.order_by(Pair.client_id1, Pair.client_id2, File1.shot_id, File2.shot_id)
    for k in q:
      retval[k[0].id] = (make_path(k[1].path, directory, extension), make_path(k[2].path, directory, extension), \
        k[0].client_id1, k[0].client_id2, k[1].id, k[2].id)
        
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
