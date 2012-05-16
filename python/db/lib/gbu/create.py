#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Fri May 11 17:20:46 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This script creates the GBU database in a single pass.
"""

import os

from .models import *
from ..utils import session


def collect_files(directory, extension = '.jpg', subdirectory = None):
  """Reads add images (in all sub-directories) of the given directory and 
  corrects the directories stored in all entries"""
  # recursively walk through the directory and collect files
  walk = [(x[0], x[1], x[2]) for x in os.walk(directory)]

  # split off the images and align the directory
  filelist = []
  dirlist = []
  for dir,subdirs,files in walk:
    filelist.extend([f for f in files if os.path.splitext(f)[1]==extension and ((not subdirectory) or subdirectory in dir)])
    dirlist.extend([dir.lstrip(directory) for f in files if os.path.splitext(f)[1]==extension and ((not subdirectory) or subdirectory in dir)])
    
  return (filelist, dirlist)




def add_files_and_protocols(session, list_dir, image_dir = None):
  """Add files (and clients) to the BANCA database."""
 
  import xml.sax
  class XmlFileReader (xml.sax.handler.ContentHandler):
    def __init__(self):
      self.m_file = File()
      self.m_file_list = []
    
    def startDocument(self):
      pass
      
    def endDocument(self):
      pass
      
    def startElement(self, name, attrs):
      if name == 'biometric-signature':
        self.m_file.m_signature = attrs['name']
      elif name == 'presentation':
        filename = attrs['file-name'] 
        self.m_file.m_filename = os.path.splitext(os.path.basename(filename))[0]
        self.m_file.m_directory = os.path.dirname(filename)
        self.m_file.m_presentation = attrs['name']
      else:
        pass

    def endElement(self, name):
      if name == 'biometric-signature':
        # assert that everything was read correctly
        assert self.m_file.m_signature and self.m_file.m_filename and self.m_file.m_presentation
        # add a file to the sessions
        self.m_file_list.append(self.m_file)
        # new file
        self.m_file = File()
      else:
        pass
      
  ################################################################################
  ##### End of XmlFileReader class ###############################################


  def read_list(xml_file, eye_file = None):
    """Reads the xml list and attaches the eye files, if given"""
    # create xml reading instance
    handler = XmlFileReader()
    xml.sax.parse(xml_file, handler)
    image_list = handler.m_file_list

    if eye_file:
      # generate temporary dictionary for faster read of the eye position file    
      image_dict={}
      for image in image_list:
        image_dict[image.m_filename] = image
      
      # read the eye position list
      f = open(eye_file)
      for line in f:
        entries=line.split(',')
        assert len(entries) == 5
        name = os.path.splitext(os.path.basename(entries[0]))[0]
        # test if these eye positions belong to any file of this list
        if name in image_dict:
          image_dict[name].eyes(entries[1:])

    return image_list
  

  def correct_dir(image_list, filenames, directories, extension = '.jpg'):
    """Iterates through the image list and corrects the directory"""
    # first, collect entries in a faster structure
    image_dict = {}
    for i in image_list:
      image_dict[i.m_filename + extension] = i
    # assert that we don't have duplicate entries
    assert len(image_dict) == len(image_list)
      
    # now, iterate through the directory list and check for the file names
    for index in range(len(filenames)):
      if filenames[index] in image_dict:
        # copy the directory of the found image
        image_dict[filenames[index]].m_directory = directories[index]
    
    # finally, do the other way around and check if every file has been found
    filenames_set = set()
    for f in filenames:
      filenames_set.add(f)
    # assert that we don't have duplicate entries
    assert len(filenames) == len(filenames_set)

    missing_files = []
    for i in image_list:
      if i.m_filename + extension not in filenames_set:
        missing_files.append(i)
        print "The image '" + i.m_filename + extension + "' was not found in the given directory"

    return missing_files


  
###########################################################################
#### Here the function really starts ######################################

  # first, read the file lists from XML files
  train_sets = Trainset.m_names 
  protocols = Protocol.m_names
  types = Protocol.m_types

  eyes_file = os.path.join(list_dir, 'alleyes.csv')
  
  train_lists = {}
  target_lists = {}
  query_lists = {}
  
  for p in train_sets:
    # Training files
    train_lists[p] = read_list(os.path.join(list_dir, 'GBU_Training_Uncontrolled%s.xml'%p), eyes_file)

  for p in protocols:
    # Target files
    target_lists[p] = read_list(os.path.join(list_dir, 'GBU_%s_Target.xml'%p), eyes_file)
    # Query files
    query_lists[p] = read_list(os.path.join(list_dir, 'GBU_%s_Query.xml'%p), eyes_file)
  all_lists = [f for f in train_lists.itervalues()]
  all_lists += [f for f in target_lists.itervalues()]
  all_lists += [f for f in query_lists.itervalues()]

  # now, correct the directories according to the real image directory structure
  if image_dir:
    print "Collecting images from directory", image_dir
    # collect all the files in the given directory
    file_list, dir_list = collect_files(image_dir)
    print "Done. Collected", len(file_list), "images."
    # correct the directories in all file lists
    for l in all_lists:
      correct_dir(l, file_list, dir_list)
      
  # Now, create file entries in the database and create clients and files
  clients = set()
  files = set()
  for l in all_lists:
    for f in l:
      if f.m_signature not in clients:
        session.add(Client(f.m_signature))
        clients.add(f.m_signature)
      if f.m_presentation not in files:
        session.add(f)
        files.add(f.m_presentation)
      
  # training sets
  for s in train_sets:
    for f in train_lists[s]:
      session.add(Trainset(s, f.m_presentation))
        
  # protocols
  for type in types:
    for protocol in protocols:
      # enroll files
      for f in target_lists[protocol]:
        session.add(Protocol(type, protocol, 'enrol', f.m_presentation))

      # probe files
      for f in query_lists[protocol]:
        session.add(Protocol(type, protocol, 'probe', f.m_presentation))

  # all right, that should be it.

def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  File.metadata.create_all(engine)
  Trainset.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)

# Driver API
# ==========

def create(args):
  """Creates or re-creates this database"""

  dbfile = args.location.replace('sqlite:///','')

  if args.recreate: 
    if args.verbose and os.path.exists(dbfile):
      print('unlinking %s...' % dbfile)
    if os.path.exists(dbfile): os.unlink(dbfile)

  if not os.path.exists(os.path.dirname(dbfile)):
    os.makedirs(os.path.dirname(dbfile))

  # the real work...
  create_tables(args)
  s = session(args.dbname, echo=args.verbose)
  add_files_and_protocols(s, args.list_directory, args.rescan_image_directory)
  s.commit()
  s.close()

def add_command(subparsers):
  """Add specific subcommands that the action "create" can use"""

  parser = subparsers.add_parser('create', help=create.__doc__)

  parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  parser.add_argument('--list-directory', metavar='DIR',
      default = "/idiap/user/mguenther/GBU_FILE_LISTS",
      help="Change the relative path to the directory containing the list of the GBU database (defaults to %(default)s)")
  parser.add_argument('--rescan-image-directory', metavar='DIR',
#      default='/idiap/resource/database/MBGC-V1',
      help="If required, select the path to the directory containing the images of the MBGC-V1 database to be re-scanned")
  
  parser.set_defaults(func=create) #action
