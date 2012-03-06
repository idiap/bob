#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Mar  5 15:38:22 CET 2012

"""
The CASIA-FASD database is a spoofing attack database which consists of three types of attacks: warped printed photographs, printed photographs with cut eyes and video attacks. The samples are taken with three types of cameras: low quality, normal quality and high quality.

References:

  1. Z. Zhang, J. Yan, S. Lei, D. Yi, S. Z. Li: "A Face Antispoofing Database with Diverse Attacks", In proceedings of the 5th IAPR International Conference on Biometrics (ICB'12), New Delhi, India, 2012."""

import os
import sys
import numpy
from .. import utils

class Database(object):

  def __init__(self):  
    self.groups = ('train', 'test')
    self.classes = ('attack', 'real')
    self.qualities = ('low','normal','high')
    self.types = ('warped', 'cut', 'video')
    self.ids = range(1, 51)

  def files(self, directory=None, extension=None, ids=[], groups=None, cls=None, qualities=None, types=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    ids
      The id of the client whose videos need to be retrieved. Should be an integer number in the range 1-50 (the total number of client is 50

    groups
      One of the protocolar subgroups of data as specified in the tuple groups, or a
      tuple with several of them.  If you set this parameter to an empty string
      or the value None, we use reset it to the default which is to get all.

    cls
      Either "attack", "real" or a combination of those (in a
      tuple). Defines the class of data to be retrieved.  If you set this
      parameter to an empty string or the value None, it will be set to the tuple ("real", "attack").

    qualities
      Either "low", "normal" or "high" or any combination of those (in a
      tuple). Defines the qualities of the videos in the database that are going to be used. If you set this
      parameter to the value None, the videos of all qualities are returned ("low", "normal", "high").

    types
      Either "warped", "cut" or "video" or any combination of those (in a
      tuple). Defines the types of attack videos in the database that are going to be used. If you set this
      parameter to the value None, the videos of all the attack types are returned ("warped", "cut", "video").

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are just pro-forma (for uniformity with the other databases).
    """

    def check_validity(l, obj, valid, default):
      """Checks validity of user input data against a set of valid values"""
      if not l: return default
      elif isinstance(l, str) or isinstance(l, int): return check_validity((l,), obj, valid, default) 
      for k in l:
        if k not in valid:
          raise RuntimeError, 'Invalid %s "%s". Valid values are %s, or lists/tuples of those' % (obj, k, valid)
      return l

    def make_path(stem, directory, extension):
      if not extension: extension = ''
      if directory: return os.path.join(directory, stem + extension)
      return stem + extension


    # check if groups set are valid
    VALID_GROUPS = self.groups
    groups = check_validity(groups, "group", VALID_GROUPS, VALID_GROUPS)

    # by default, do NOT grab enrollment data from the database
    VALID_CLASSES = self.classes
    VALID_TYPES = self.types
    if cls == None and types != None: # types are strictly specified which means we don't need the calss of real accesses
      cls = ('attack',)
    else:
      cls = check_validity(cls, "class", VALID_CLASSES, ('real', 'attack'))

    # check if video quality types are valid
    VALID_QUALITIES = self.qualities
    qualities = check_validity(qualities, "quality", VALID_QUALITIES, VALID_QUALITIES)

    # check if attack types are valid

    if cls != ('real',): # if the class is 'real' only, then there is no need for types to be reset to the default (real accesses have no types)
      types = check_validity(types, "type", VALID_TYPES, VALID_TYPES)
  
    VALID_IDS = self.ids
    ids = check_validity(ids, "id", VALID_IDS, VALID_IDS)

    retval = {}
    key = 0
    
    db_mappings = {'real_normal':'1', 'real_low':'2', 'real_high':'HR_1', 'warped_normal':'3', 'warped_low':'4', 'warped_high':'HR_2', 'cut_normal':'5', 'cut_low':'6', 'cut_high':'HR_3', 'video_normal':'7', 'video_low':'8', 'video_high':'HR_4'}    

    # identitites in the training set are assigned ids 1-20, identities in the test set are assigned ids 21-50
    for i in ids:
      for g in groups:
        if (g == 'train' and i > 20) or (g == 'test' and i <= 20): continue;
        cur_id = i 
        if g == 'test': cur_id = i - 20; # the id within the group subset
        folder_name = g + '_release'
        for q in qualities:
          if cls == ('real',) and types != None: continue; # category real + any type does not exist 
          for c in cls:          
            if c == 'real': # the class real doesn't have any different types, only the attacks can be of different type
              name = os.path.join(folder_name, "%d" % cur_id, db_mappings[c + '_' + q])
              retval[key] = make_path(name, directory, extension)
              key = key + 1  
            else:  
              for t in types:
                name = os.path.join(folder_name, "%d" % cur_id, db_mappings[t + '_' + q])
                retval[key] = make_path(name, directory, extension)
                key = key + 1  
              
    return retval
              

  def save_one(self, filename, obj, directory, extension):
    """Saves a single object supporting the bob save() protocol.

    This method will call save() on the the given object using the correct
    database filename stem for the given id.
    
    Keyword Parameters:

    filename
      The unique filename under which the object will be saved. Before calling this method, the method files() should be called (with no directory and extension arguments passed) in order to obtain the unique filenames for each of the files to be saved.

    obj
      The object that needs to be saved, respecting the bob save() protocol.

    directory
      This is the base directory to which you want to save the data. The
      directory is tested for existence and created if it is not there with
      os.makedirs()

    extension
      The extension determines the way each of the arrays will be saved.
    """

    from ...io import save

    fullpath = os.path.join(directory, filename + extension)
    fulldir = os.path.dirname(fullpath)
    utils.makedirs_safe(fulldir)
    save(obj, fullpath)
