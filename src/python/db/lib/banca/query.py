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

  def __group_replace_alias__(self, l):
    """Replace 'dev' by 'g1' and 'eval' by 'g2' in a list of groups, and 
       returns the new list"""
    if not l: return l
    elif isinstance(l, str): return self.__group_replace_alias__((l,))
    l2 = []
    for val in l:
      if(val == 'dev'): l2.append('g1')
      elif(val == 'eval'): l2.append('g2')
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

  def clients(self, protocol=None, groups=None, gender=None, language=None, subworld=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua")

    groups
      The groups to which the clients belong ("g1", "g2", "world").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    gender
      The genders to which the clients belong ("f", "m")

    language
      TODO: only English is currently supported
      The language spoken by the clients ("en",)

    subworld
      Specify a split of the world data ("onethird", "twothirds", "")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    Returns: A list containing all the client ids which have the given
    properties.
    """

    groups = self.__group_replace_alias__(groups)
    VALID_GROUPS = ('g1', 'g2', 'world')
    VALID_GENDERS = ('m', 'f')
    VALID_LANGUAGES = ('en',)
    VALID_SUBWORLDS = ('onethird', 'twothirds')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    gender = self.__check_validity__(gender, "gender", VALID_GENDERS)
    language = self.__check_validity__(language, "language", VALID_LANGUAGES)
    subworld = self.__check_validity__(subworld, "subworld", VALID_SUBWORLDS)

    retval = []
    # List of the clients
    if "world" in groups:
      if len(subworld)==1:
        q = self.session.query(Client).join(Subworld).filter(Subworld.name.in_(subworld))
      else:
        q = self.session.query(Client).filter(Client.sgroup == 'world')
      q = q.filter(Client.gender.in_(gender)).\
            filter(Client.language.in_(language)).\
          order_by(Client.id)
      for id in [k.id for k in q]: 
        retval.append(id)

    if 'g1' in groups or 'g2' in groups:
      q = self.session.query(Client).filter(Client.sgroup != 'world').\
            filter(Client.sgroup.in_(groups)).\
            filter(Client.gender.in_(gender)).\
            filter(Client.language.in_(language)).\
            order_by(Client.id)
      for id in [k.id for k in q]: 
        retval.append(id)

    return retval

  def Tclients(self, protocol=None, groups=None):
    """Returns a set of T-Norm clients for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").
    
    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    Returns: A list containing all the model ids belonging to the given group.
    """

    groups = self.__group_replace_alias__(groups)
    VALID_GROUPS = ('g1', 'g2')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    tgroups = []
    if 'g1' in groups:
      tgroups.append('g2')
    if 'g2' in groups:
      tgroups.append('g1')
    return self.clients(protocol, tgroups)

  def Zclients(self, protocol=None, groups=None):
    """Returns a set of Z-Norm clients for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").
    
    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    Returns: A list containing all the model ids belonging to the given group.
    """

    groups = self.__group_replace_alias__(groups)
    VALID_GROUPS = ('g1', 'g2')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    zgroups = []
    if 'g1' in groups:
      zgroups.append('g2')
    if 'g2' in groups:
      zgroups.append('g1')
    return self.clients(protocol, zgroups)


  def models(self, protocol=None, groups=None):
    """Returns a set of models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").
    
    groups
      The groups to which the subjects attached to the models belong ("g1", "g2", "world")
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.clients(protocol, groups)

  def Tmodels(self, protocol=None, groups=None):
    """Returns a set of T-Norm models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").
    
    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.Tclients(protocol, groups)

  def Zmodels(self, protocol=None, groups=None):
    """Returns a set of Z-Norm models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").
    
    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

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
      return q.first().real_id

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
      purposes=None, model_ids=None, groups=None, classes=None, 
      languages=None, subworld=None):
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
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    subworld
      Specify a split of the world data ("onethird", "twothirds", "")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    groups = self.__group_replace_alias__(groups)
    VALID_PROTOCOLS = ('Mc', 'Md', 'Ma', 'Ud', 'Ua', 'P', 'G')
    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('g1', 'g2', 'world')
    VALID_LANGUAGES = ('en', 'fr', 'sp')
    VALID_CLASSES = ('client', 'impostor')
    VALID_SUBWORLDS = ('onethird', 'twothirds')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    languages = self.__check_validity__(languages, "language", VALID_LANGUAGES)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)
    subworld = self.__check_validity__(subworld, "subworld", VALID_SUBWORLDS)

    retval = {}

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
    
    if 'world' in groups:
      if len(subworld) == 1:
        q = self.session.query(File).join(Client).join(Subworld).filter(Subworld.name.in_(subworld))
      else:
        q = self.session.query(File).join(Client)
      q = q.filter(Client.sgroup == 'world').\
            filter(Client.language.in_(languages))
      if model_ids:
        q = q.filter(File.real_id.in_(model_ids))
      q = q.order_by(File.real_id, File.session_id, File.claimed_id, File.shot) 
      for k in q:
        retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)
    
    if ('g1' in groups or 'g2' in groups):
      if('enrol' in purposes):
        q = self.session.query(File).join(Client).join(Session).join(Protocol).\
              filter(File.claimed_id == File.real_id).\
              filter(Client.sgroup.in_(groups)).\
              filter(Client.language.in_(languages)).\
              filter(Protocol.name.in_(protocol)).\
              filter(Protocol.purpose == 'enrol')
        if model_ids:
          q = q.filter(File.claimed_id.in_(model_ids))
        q = q.order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
        for k in q:
          retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)
      if('probe' in purposes):
        if('client' in classes):
          q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                filter(File.claimed_id == File.real_id).\
                filter(Client.sgroup.in_(groups)).\
                filter(Client.language.in_(languages)).\
                filter(Protocol.name.in_(protocol)).\
                filter(Protocol.purpose == 'probe')
          if model_ids:
            q = q.filter(File.claimed_id.in_(model_ids))
          q = q.order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
          for k in q:
            retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)
        if('impostor' in classes):
          q = self.session.query(File).join(Client).join(Session).join(Protocol).\
                filter(File.claimed_id != File.real_id).\
                filter(Client.sgroup.in_(groups)).\
                filter(Client.language.in_(languages)).\
                filter(Protocol.name.in_(protocol)).\
                filter(or_(Protocol.purpose == 'probe', Protocol.purpose == 'probeImpostor'))
          if model_ids:
            q = q.filter(File.claimed_id.in_(model_ids))
          for k in q:
            retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)

    return retval

  def files(self, directory=None, extension=None, protocol=None,
      purposes=None, model_ids=None, groups=None, classes=None, 
      languages=None, subworld=None):
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
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    subworld
      Specify a split of the world data ("onethird", "twothirds", "")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. Clients from other groups ("dev", "eval")
      will in this case be ignored.

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the BANCA database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    retval = {}
    d = self.objects(directory, extension, protocol, purposes, model_ids, groups, classes, languages, subworld)
    for k in d: retval[k] = d[k][0]

    return retval


  def Tobjects(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, languages=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    groups = self.__group_replace_alias__(groups)
    VALID_GROUPS = ('g1', 'g2')
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    tgroups = []
    if 'g1' in groups:
      tgroups.append('g2')
    if 'g2' in groups:
      tgroups.append('g1')
    return self.objects(directory, extension, protocol, 'enrol', model_ids, tgroups, 'client', languages)

  def Tfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, languages=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.Tobjects(directory, extension, protocol, model_ids, groups, languages)
    for k in d: retval[k] = d[k][0]

    return retval


  def Zobjects(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, languages=None):
    """Returns a set of filenames to perform Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    groups = self.__group_replace_alias__(groups)
    VALID_PROTOCOLS = ('Mc', 'Md', 'Ma', 'Ud', 'Ua', 'P', 'G')
    VALID_GROUPS = ('g1', 'g2')
    VALID_LANGUAGES = ('en', 'fr', 'sp')

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    languages = self.__check_validity__(languages, "language", VALID_LANGUAGES)
    zgroups = []
    if 'g1' in groups:
      zgroups.append('g2')
    if 'g2' in groups:
      zgroups.append('g1')

    retval = {}

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
 
    # Files used as client probes
    q = self.session.query(File).join(Client).join(Session).join(Protocol).\
            filter(File.claimed_id == File.real_id).\
            filter(Client.sgroup.in_(zgroups)).\
            filter(Client.language.in_(languages)).\
            filter(Protocol.name.in_(protocol)).\
            filter(Protocol.purpose == 'probe')
    if model_ids:
      q = q.filter(File.real_id.in_(model_ids))
    q = q.order_by(File.claimed_id, File.session_id, File.real_id, File.shot)
    for k in q:
      retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)

    # Files used as impostor probes
    q = self.session.query(File).join(Client).join(Session).join(Protocol).\
            filter(File.claimed_id != File.real_id).\
            filter(Client.sgroup.in_(zgroups)).\
            filter(Client.language.in_(languages)).\
            filter(Protocol.name.in_(protocol)).\
            filter(or_(Protocol.purpose == 'probe', Protocol.purpose == 'probeImpostor'))
    if model_ids:
      q = q.filter(File.real_id.in_(model_ids))
    for k in q:
      retval[k.id] = (make_path(k.path, directory, extension), k.claimed_id, k.claimed_id, k.real_id, k.path)

    return retval

  def Zfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, languages=None):
    """Returns a set of filenames to perform Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the BANCA protocols ("P", "G", "Mc", "Md", "Ma", "Ud", "Ua").

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      The groups to which the clients belong ("g1", "g2").
      Note that 'dev' is an alias to 'g1' and 'test' an alias to 'g2'

    languages
      The language spoken by the clients ("en")
      TODO: only English is currently supported
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model
      - 3: the real id
      - 4: the "stem" path (basename of the file)
    considering allthe filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.Zobjects(directory, extension, protocol, model_ids, groups, languages)
    for k in d: retval[k] = d[k][0]

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
    utils.makedirs_safe(fulldir)
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
