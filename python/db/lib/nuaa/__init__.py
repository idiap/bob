#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Wed Feb 15 12:57:28 CET 2012

"""
The NUAA database is a spoofing attack database which consists of real accesses and only printed photo attacks. There are three versions of the database: version composed of raw images, version composed of the cropped faces detected in the images and version composed of 64x64 normalized faces detected in the images. There are only train and test set defined.

References:

  1. X. Tan, Y. Li, J. Liu, L. Jiang: "Face Liveness Detection from a Single Image with Sparse Low Rank Bilinear Discriminative Model" Fisher,R.A. "The use of multiple measurements in taxonomic problems", Proceedings of 11th European Conference on Computer Vision (ECCV'10), Crete, Greece, September 2010; p.9-11
"""

import os
import sys
import numpy
from .. import utils

class Database(object):

  def __init__(self):  
    self.groups = ('train', 'test')
    self.classes = ('attack', 'real')
    self.versions = ('raw','detected_face','normalized_face')

  def files(self, directory=None, extension=None, groups=None, cls=None, versions=None):
    """Returns a set of filenames for the specific query by the user.

    Keyword Parameters:

    directory
      A directory name that will be prepended to the final filepath returned

    extension
      A filename extension that will be appended to the final filepath returned

    groups
      One of the protocolar subgroups of data as specified in the tuple groups, or a
      tuple with several of them.  If you set this parameter to an empty string
      or the value None, we use reset it to the default which is to get all.

    cls
      Either "attack", "real" or any combination of those (in a
      tuple). Defines the class of data to be retrieved.  If you set this
      parameter to an empty string or the value None, we use reset it to the
      default, ("real", "attack").

    versions
      Either "raw", "detected_face", "normalized_face" or any combination of those (in a
      tuple). Defines the version of the database that is going to be used. If you set this
      parameter to the value None, the images from all the versions are returned ("raw", "detected_face", "normalized_face").

    Returns: A dictionary containing the resolved filenames considering all
    the filtering criteria. The keys of the dictionary are just pro-forma (for uniformity with the other databases).
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


    # check if groups set are valid
    VALID_GROUPS = self.groups
    groups = check_validity(groups, "group", VALID_GROUPS, VALID_GROUPS)

    # check if supports set are valid
    VALID_VERSIONS = self.versions
    versions = check_validity(versions, "version", VALID_VERSIONS, VALID_VERSIONS)

    # by default, do NOT grab enrollment data from the database
    VALID_CLASSES = self.classes
    cls = check_validity(cls, "class", VALID_CLASSES, ('real', 'attack'))
  
    retval = {}
    key = 0
    
    # because of couple of non-uniformities in the naming system in the database, we need to introduce these dictionaries to convert between names
    version_dict = {'raw':'raw', 'detected_face':'Detectedface', 'normalized_face':'NormalizedFace'}
    version_dict_1 = {'raw':'Raw', 'detected_face':'Face', 'normalized_face':'Normalized'}
    cls_dict = {'attack':'Imposter', 'real':'Client'}

    for v in versions:   
      if (v == 'raw' or v == 'detected_face') and extension=='.bmp': extension = '.jpg' # the extension is .jpg for raw data and detected faces images
      if v == 'normalized_face' and extension=='.jpg': extension = '.bmp' # the extension is .bmp for normalized faces images
      for c in cls:
        for g in groups:
          bdir = version_dict[v]
          # the filename with the list of files belonging to the user specified criteria
          readfilename = os.path.join(os.path.dirname(__file__), bdir, cls_dict[c].lower()+'_'+g+'_'+version_dict_1[v].lower()+'.txt')
          readfilelines = open(readfilename, 'r').readlines()
          filesdir = os.path.join(bdir, cls_dict[c]+version_dict_1[v]) # the directory where the files are stored
          for i in readfilelines:
            name = i.partition('.')[0].replace("\\", "/") # remove the initial extension, do string procesing
            retval[key] = make_path(os.path.join(filesdir, name), directory, extension)
            key = key + 1
    return retval

  def filter_files(self, filenames, client_no=None, glasses=None, conditions=None, session=None):
    """ Filters the filenames in a dictionary and returns a filtered dictionary which contains only the images with the specified criteria.

    Keyword Parameters:

    filenames
      A dictionary with filenames (most probably obtained using the files() method).

    client_no 
      The number of the client. A string (or tuple of strings) with values from '0001'-'0016'

    glasses
      A string (or tuple of strings) with value '00' for clients with glasses and '01' for cleints without glasses

    conditions
      A string (or tuple of strings) with values '00'-'08' for various combinations of lighting conditions and spoofing images poses

    session
      A string (or tuple of strings) with values '01'-'03' for three different client enrollment sessions
    """
    retval = {}
    newkey = 0
    for key, filename in filenames.items():
      short_filename = filename.rpartition('/')[2] # just the filename (without the full path)
      stems = short_filename.split('_')
      if client_no != None:
        if stems[0] not in client_no:
          continue
      if glasses != None:
        if stems[1] not in glasses:
          continue
      if conditions != None:
        if stems[2] not in conditions:
          continue
      if session != None:
        if stems[3] not in session:
          continue
      newkey = newkey + 1
      retval[newkey] = filename
        
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
