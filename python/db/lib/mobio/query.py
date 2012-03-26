#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This module provides the Dataset interface allowing the user to query the
MOBIO database in the most obvious ways.
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

  def __gender_replace__(self,l):
    """Replace 'female' by 'f' and 'male' by 'm', and returns the new list"""
    if not l: return l
    elif isinstance(l, str): return self.__gender_replace__((l,))
    l2 = []
    for val in l:
      if(val == 'female'): l2.append('f')
      elif(val == 'male'): l2.append('m')
      else: l2.append(val)
    return tuple(l2)


  def clients(self, protocol=None, groups=None, subworld=None, gender=None):
    """Returns a set of clients for the specific query by the user.

    Keyword Parameters:

    protocol
      The protocol to consider ('male', 'female')

    groups
      The groups to which the clients belong ('dev', 'eval', 'world')
      Please note that world data are protocol/gender independent

    subworld
      Specify a split of the world data ("twothirds", "")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    gender
      The gender to consider ('male', 'female')

    Returns: A list containing all the client ids which have the given
    properties.
    """

    VALID_PROTOCOLS = ('female', 'male')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_SUBWORLDS = ('onethird','twothirds',)
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    protocol = self.__gender_replace__(protocol)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    subworld = self.__check_validity__(subworld, "subworld", VALID_SUBWORLDS)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    retval = []
    # World data (gender independent)
    if "world" in groups:
      if len(subworld)==1:
        q = self.session.query(Client).join(SubworldClient).filter(SubworldClient.name.in_(subworld))
      else:
        q = self.session.query(Client).filter(Client.sgroup == 'world')
      if gender:
        q = q.filter(Client.gender.in_(gender))
      q = q.order_by(Client.id)
      for id in [k.id for k in q]: 
        retval.append(id)
    
    # dev / eval data
    if 'dev' in groups or 'eval' in groups:
      q = self.session.query(Client).filter(and_(Client.sgroup != 'world', Client.sgroup.in_(groups)))
      if protocol:
        q = q.filter(Client.gender.in_(protocol))
      if gender:
        q = q.filter(Client.gender.in_(gender))
      q = q.order_by(Client.id)
      for id in [k.id for k in q]: 
        retval.append(id)

    return retval

  def tclients(self, protocol=None, groups=None, gender=None):
    """Returns a set of T-Norm clients for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the MOBIO protocols ("male", "female"). T-Norm clients are gender
      independent by default for MOBIO.
    
    groups
      The groups to which the clients belong ("dev", "eval").
      Useless as they are independent from 'dev' and 'eval' for this database

    gender
      The gender to consider ('male', 'female')

    Returns: A list containing all the client ids belonging to the given group.
    """

    VALID_PROTOCOLS = ('female', 'male')
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    protocol = self.__gender_replace__(protocol)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    q = self.session.query(TModel).join(Client)
    if gender:
      q = q.filter(Client.gender.in_(gender))
    q = q.order_by(TModel.client_id)
    
    tclient = []
    for cid in [k.client_id for k in q]:
      if not cid in tclient: tclient.append(cid)
    return tclient

  def zclients(self, protocol=None, groups=None, gender=None):
    """Returns a set of Z-Norm clients for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the MOBIO protocols ("male", "female").
    
    groups
      The groups to which the clients belong ("dev", "eval").
      Useless as they are independent from 'dev' and 'eval' for this database

    gender
      The gender to consider ('male', 'female')

    Returns: A list containing all the client ids belonging to the given group.
    """

    VALID_PROTOCOLS = ('female', 'male')
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    protocol = self.__gender_replace__(protocol)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    q = self.session.query(ZClient).join(Client)
    if gender:
      q = q.filter(Client.gender.in_(gender))
    q = q.order_by(ZClient.client_id)     

    zclient = []
    for cid in [k.client_id for k in q]:
      if not cid in zclient: zclient.append(cid)
    return zclient


  def models(self, protocol=None, groups=None, subworld=None, gender=None):
    """Returns a set of models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the Mobio protocols ("male", "female").

    groups
      The groups to which the subjects attached to the models belong ("dev", "eval", "world")
      Please note that world data are protocol/gender independent

    gender
      The gender to consider ('male', 'female')

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.clients(protocol, groups, subworld, gender)


  def tmodels(self, protocol=None, groups=None, gender=None):
    """Returns a set of T-Norm models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the Mobio protocols ("male", "female").
    
    groups
      The groups to which the clients belong ("dev", "eval").
      Useless as they are independent from 'dev' and 'eval' for this database

    gender
      The gender to consider ('male', 'female')

    Returns: A list containing all the model ids belonging to the given group.
    """

    VALID_PROTOCOLS = ('female', 'male')
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    protocol = self.__gender_replace__(protocol)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    tmodel = []
    q = self.session.query(TModel).join(Client)
    if gender:
      q = q.filter(Client.gender.in_(gender))
    q = q.order_by(TModel.id)     
    for tid in [k.id for k in q]:
      if not tid in tmodel: tmodel.append(tid)
    return tmodel

  def zmodels(self, protocol=None, groups=None, gender=None):
    """Returns a set of Z-Norm models for the specific query by the user.

    Keyword Parameters:

    protocol
      One of the Mobio protocols ("male", "female").
    
    groups
      The groups to which the clients belong ("dev", "eval").
      Useless as they are independent from 'dev' and 'eval' for this database

    gender
      The gender to consider ('male', 'female')

    Returns: A list containing all the model ids belonging to the given group.
    """

    return self.zclients(protocol, groups, gender)

  def get_client_id_from_model_id(self, model_id):
    """Returns the client_id attached to the given model_id
    
    Keyword Parameters:

    model_id
      The model_id to consider

    Returns: The client_id attached to the given model_id
    """
    return model_id

  def get_client_id_from_tmodel_id(self, model_id):
    """Returns the client_id attached to the given T-Norm model_id
    
    Keyword Parameters:

    model_id
      The model_id to consider

    Returns: The client_id attached to the given T-Norm model_id
    """

    q = self.session.query(TModel).filter(TModel.id == model_id)
    q = q.order_by(TModel.id)     
    if q.count() !=1:
      #throw exception?
      return None
    else:
      return q.first().client_id

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
      return q.first().client_id

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
      purposes=None, model_ids=None, groups=None, classes=None,
      subworld=None, gender=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the MOBIO protocols ('male', 'female').

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
      One of the groups ('dev', 'eval', 'world') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    subworld
      Specify a split of the world data ("twothirds", "")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    gender
      The gender to consider ('male', 'female')

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id (only valid if len(model_ids == 1))
      - 2: the claimed id attached to the model (only valid if len(model_ids == 1))
      - 3: the real id
      - 4: the "stem" path (basename of the file)

    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('female', 'male')
    VALID_PURPOSES = ('enrol', 'probe')
    VALID_GROUPS = ('dev', 'eval', 'world')
    VALID_CLASSES = ('client', 'impostor')
    VALID_SUBWORLDS = ('onethird', 'twothirds',)

    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    purposes = self.__check_validity__(purposes, "purpose", VALID_PURPOSES)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    classes = self.__check_validity__(classes, "class", VALID_CLASSES)
    subworld = self.__check_validity__(subworld, "subworld", VALID_SUBWORLDS)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    retval = {}    
    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)

    if 'world' in groups:
      q = self.session.query(File).join(Client).\
            filter(Client.sgroup == 'world')
      if model_ids:
        q = q.filter(Client.id.in_(model_ids))
      if subworld and len(subworld)==1:
        q = q.join(SubworldFile).\
              filter(SubworldFile.name.in_(subworld)).\
              filter(and_(File.client_id == SubworldFile.client_id, File.session_id == SubworldFile.session_id,
                          File.speech_type == SubworldFile.speech_type, File.shot_id == SubworldFile.shot_id))
      if gender:
        q = q.filter(Client.gender.in_(gender))
      q = q.order_by(File.client_id, File.session_id, File.speech_type, File.shot_id, File.device)
      for k in q:
        retval[k.id] = (make_path(k.path, directory, extension), k.client_id, k.client_id, k.client_id, k.path)
    
    if ('dev' in groups or 'eval' in groups):
      if('enrol' in purposes):
        q = self.session.query(File, Protocol, ProtocolEnrolSession).join(Client).\
              filter(Client.sgroup.in_(groups)).\
              filter(and_(File.client_id == ProtocolEnrolSession.client_id, File.session_id == ProtocolEnrolSession.session_id)).\
              filter(Protocol.name == ProtocolEnrolSession.name).\
              filter(and_(Protocol.name.in_(protocol), Protocol.purpose == 'enrol', File.speech_type == Protocol.speech_type))
        if model_ids:
          q = q.filter(Client.id.in_(model_ids))
        if gender:
          q = q.filter(Client.gender.in_(gender))
        q = q.order_by(File.client_id, File.session_id, File.speech_type, File.shot_id, File.device)
        for k in q:
          retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)
      if('probe' in purposes):
        if('client' in classes):
          q = self.session.query(File, Protocol, ProtocolEnrolSession).join(Client).\
                filter(Client.sgroup.in_(groups)).\
                filter(and_(File.client_id == ProtocolEnrolSession.client_id, File.session_id != ProtocolEnrolSession.session_id)).\
                filter(Protocol.name == ProtocolEnrolSession.name).\
                filter(and_(Protocol.name.in_(protocol), Protocol.purpose == 'probe', File.speech_type == Protocol.speech_type))
          if model_ids:
            q = q.filter(Client.id.in_(model_ids))
          if gender:
            q = q.filter(Client.gender.in_(gender))
          q = q.order_by(File.client_id, File.session_id, File.speech_type, File.shot_id, File.device)
          for k in q:
            retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)
        if('impostor' in classes):
          q = self.session.query(File, Protocol, ProtocolEnrolSession).join(Client).\
                filter(Client.sgroup.in_(groups)).\
                filter(and_(File.client_id == ProtocolEnrolSession.client_id, File.session_id != ProtocolEnrolSession.session_id)).\
                filter(Protocol.name == ProtocolEnrolSession.name).\
                filter(and_(Protocol.name.in_(protocol), Protocol.purpose == 'probe', File.speech_type == Protocol.speech_type))
          if(model_ids and len(model_ids)==1):
            q = q.filter(not_(Client.id.in_(model_ids)))
          if gender:
            q = q.filter(Client.gender.in_(gender))
          q = q.order_by(File.client_id, File.session_id, File.speech_type, File.shot_id, File.device)
          for k in q:
            if(model_ids and len(model_ids) == 1):
              retval[k[0].id] = (make_path(k[0].path, directory, extension), model_ids[0], model_ids[0], k[0].client_id, k[0].path)
            else:
              retval[k[0].id] = (make_path(k[0].path, directory, extension), k[0].client_id, k[0].client_id, k[0].client_id, k[0].path)
        
    return retval

  def files(self, directory=None, extension=None, protocol=None,
      purposes=None, model_ids=None, groups=None, classes=None,
      subworld=None, gender=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the MOBIO protocols ('male', 'female').

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
      One of the groups ('dev', 'eval', 'world') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    classes
      The classes (types of accesses) to be retrieved ('client', 'impostor') 
      or a tuple with several of them. If 'None' is given (this is the 
      default), it is considered the same as a tuple with all possible values.

    subworld
      Specify a split of the world data ("twothirds", "")
      In order to be considered, "world" should be in groups and only one 
      split should be specified. 

    gender
      The gender to consider ('male', 'female')

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are unique identities 
    for each file in the Biosecure database. Conserve these numbers if you 
    wish to save processing results later on.
    """

    retval = {}
    d = self.objects(directory, extension, protocol, purposes, model_ids, 
      groups, classes, subworld, gender)
    for k in d: retval[k] = d[k][0]

    return retval


  def tobjects(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, gender=None):
    """Returns a set of filenames for enroling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the MOBIO protocols ('male', 'female').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ('dev', 'eval', 'world') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    gender
      The gender to consider ('male', 'female')

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id
      - 2: the claimed id attached to the model 
      - 3: the real id
      - 4: the "stem" path (basename of the file)

    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('female', 'male')
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    retval = {}    
    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)

    q = self.session.query(File).join(Client).join(TModel).\
          filter(and_(File.client_id == TModel.client_id, File.session_id == TModel.session_id,
                      File.speech_type == TModel.speech_type))
    if model_ids:
      q = q.filter(TModel.id.in_(model_ids))
    if gender:
      q = q.filter(Client.gender.in_(gender))
    q = q.order_by(File.client_id, File.session_id, File.speech_type, File.shot_id, File.device)
    for k in q:
      retval[k.id] = (make_path(k.path, directory, extension), k.client_id, k.client_id, k.client_id, k.path) 
    return retval

  def tfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, gender=None):
    """Returns a set of filenames for enrolling T-norm models for score 
       normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the MOBIO protocols ('male', 'female').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ('dev', 'eval', 'world') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    gender
      The gender to consider ('male', 'female')

    Returns: A list of filenames
    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the BANCA database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.tobjects(directory, extension, protocol, model_ids, groups, 
      gender)
    for k in d: retval[k] = d[k][0]

    return retval


  def zobjects(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, gender=None):
    """Returns a set of filenames to perform Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the MOBIO protocols ('male', 'female').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ('dev', 'eval', 'world') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    gender
      The gender to consider ('male', 'female')

    Returns: A dictionary containing:
      - 0: the resolved filenames 
      - 1: the model id # not applicable in this case
      - 2: the claimed id attached to the model # not applicable in this case
      - 3: the real id
      - 4: the "stem" path (basename of the file)

    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the MOBIO database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    def make_path(stem, directory, extension):
      import os
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension

    VALID_PROTOCOLS = ('female', 'male')
    VALID_GROUPS = ('dev', 'eval', 'world')
    protocol = self.__check_validity__(protocol, "protocol", VALID_PROTOCOLS)
    groups = self.__check_validity__(groups, "group", VALID_GROUPS)
    gender = self.__check_validity__(gender, "gender", VALID_PROTOCOLS)
    gender = self.__gender_replace__(gender)

    retval = {}

    if(isinstance(model_ids,str)):
      model_ids = (model_ids,)
 
    # Files used as impostor probes (all the samples from the Z-Norm clients)
    q = self.session.query(File).join(Client).join(ZClient).\
            filter(File.client_id == ZClient.client_id)
    if model_ids:
      q = q.filter(File.client_id.in_(model_ids))
    if gender:
      q = q.filter(Client.gender.in_(gender))
    q = q.order_by(File.client_id, File.session_id, File.speech_type, File.shot_id)
    for k in q:
      retval[k.id] = (make_path(k.path, directory, extension), k.client_id, k.client_id, k.client_id, k.path)

    return retval

  def zfiles(self, directory=None, extension=None, protocol=None,
      model_ids=None, groups=None, gender=None):
    """Returns a set of filenames to perform Z-norm score normalization.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    protocol
      One of the MOBIO protocols ('male', 'female').

    model_ids
      Only retrieves the files for the provided list of model ids (claimed 
      client id).  If 'None' is given (this is the default), no filter over 
      the model_ids is performed.

    groups
      One of the groups ('dev', 'eval', 'world') or a tuple with several of them. 
      If 'None' is given (this is the default), it is considered the same as a 
      tuple with all possible values.

    gender
      The gender to consider ('male', 'female')

    Returns: A list of filenames
    considering all the filtering criteria. The keys of the dictionary are 
    unique identities for each file in the MOBIO database. Conserve these 
    numbers if you wish to save processing results later on.
    """

    retval = {}
    d = self.zobjects(directory, extension, protocol, model_ids, groups, 
      gender)
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
