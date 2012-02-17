#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This script creates the Multi-PIE database in a single pass.
"""

import os
import fileinput

from .models import *
from ..utils import session

def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_clients(session, filelist):
  """Add files (and clients) to the Multi-PIE database."""

  # Define development and evaluation set in term of client ids
  dev_ids =   [2, 4, 6, 8, 10, 15, 18, 20, 22, 27, 33, 35, 38, 40, 42, 46, 48, 50, 52, 54, 57, 64, 68, 69, 71, 78, 80, 85, 97, 102, 
              105, 107, 110, 111, 115, 118, 123, 125, 126, 128, 132, 137, 139, 143, 149, 157, 167, 169, 170, 177, 184, 186, 190, 191, 
              193, 198, 202, 205, 208, 220, 227, 235, 241, 248]

  eval_ids =  [3, 5, 9, 11, 14, 17, 19, 23, 28, 29, 34, 36, 41, 43, 44, 47, 49, 53, 55, 56, 62, 67, 70, 74, 76, 79, 83, 100, 103, 104,
              106, 108, 112, 116, 117, 122, 124, 127, 129, 131, 133, 138, 145, 150, 156, 161, 168, 175, 178, 181, 185, 188, 192, 194, 
              196, 199, 203, 209, 223, 225, 230, 236, 240, 246, 250]
 
  def add_client(session, client_string, client_dict):
    """Parse a single client string and add its content to the database.
       Also add a client entry if not already in the database."""

    v = client_string.split(' ')
    if (v[2] == 'Male'):
      v[2] = 'm'
    elif (v[2] == 'Female'):
      v[2] = 'f'
    v[6] = v[6].rstrip() # chomp new line
    first_session = 0
    second_session = 0
    third_session = 0
    fourth_session = 0
    if(v[3] == '1'): 
      first_session = 1
      if(v[4] == '1'):
        second_session = 2
        if(v[5] == '1'):
          third_session = 3
          if(v[6] == '1'):
            fourth_session = 4
        elif(v[6] == '1'):
          third_session = 4
      elif(v[5] == '1'):
        second_session = 3
        if(v[6] == '1'):
          third_session = 4
      elif(v[6] == '1'):
        second_session = 4
    elif(v[4] == '1'): 
      first_session = 2
      if(v[5] == '1'):
        second_session = 3
        if(v[6] == '1'):
          third_session = 4
      elif(v[6] == '1'):
        second_session = 4
    elif(v[5] == '1'): 
      first_session = 3
      if(v[6] == '1'):
        second_session = 4
    elif(v[6] == '1'): 
      first_session = 4
    #TODO: if first_session == 0: raises an error

    if not (v[0] in client_dict):
      group = 'world'
      if int(v[0]) in dev_ids: group = 'dev'
      elif int(v[0]) in eval_ids: group = 'eval'
        
      session.add(Client(int(v[0]), group, int(v[1]), v[2], first_session, second_session, third_session, fourth_session))
      client_dict[v[0]] = True
  
  client_dict = {} 
  for line in fileinput.input(filelist):
    add_client(session, line, client_dict)

def add_files(session, imagedir, all_poses):
  """Add files (and clients) to the Multi-PIE database."""
  
  def add_mv_file(session, filename, session_id, client_id, recording_id, camera_id):
    """Parse a single filename and add it to the list.
       Also add a client entry if not already in the database."""
    v = os.path.splitext(filename)[0].split('_')
    f = File(int(client_id), filename, int(session_id[8]), int(recording_id), 'multiview')
    session.add(f)
    # We want to make use of the new assigned file id
    # We need to do the following:
    session.flush()
    session.refresh(f)
    session.add(FileMultiview(f.id, camera_id, int(v[5])))
 
  def add_hr_file(session, filename, session_id, client_id):
    """Parse a single filename and add it to the list.
       Also add a client entry if not already in the database."""
    v = os.path.splitext(filename)[0].split('_')
    session.add(File(int(client_id), filename, int(session_id[8]), int(v[1]), 'highres'))
 
  list_of_files = {}
  # session
  for session_id in filter(nodot, os.listdir(imagedir)):
    se_dir = os.path.join(imagedir, session_id)

    # multiview
    mv_dir = os.path.join(se_dir, 'multiview')
    # client id
    for client_id in filter(nodot, os.listdir(mv_dir)):
      client_dir = os.path.join(mv_dir, client_id)
      # recording id
      for recording_id in filter(nodot, os.listdir(client_dir)):
        recording_dir = os.path.join(client_dir, recording_id)
        # camera id
        for camera_id in filter(nodot, os.listdir(recording_dir)):
          # Check if it is the frontal camera 05_1
          if ((not all_poses) and camera_id != '05_1'):
            continue
          camera_dir = os.path.join(recording_dir, camera_id)
          # flashes/images
          for filename in filter(nodot, os.listdir(camera_dir)):
            basename, extension = os.path.splitext(filename)
            add_mv_file(session, os.path.join( session_id, 'multiview', client_id, recording_id, camera_id, basename), session_id, client_id, recording_id, camera_id)

    # highres
    hr_dir = os.path.join(se_dir, 'highres')
    # client id
    for client_id in filter(nodot, os.listdir(hr_dir)):
      client_dir = os.path.join(hr_dir, client_id)
      # flashes/images
      for filename in filter(nodot, os.listdir(client_dir)):
        basename, extension = os.path.splitext(filename)
        add_hr_file(session, os.path.join( session_id, 'highres', client_id, basename), session_id, client_id)

def add_protocols(session):
  """Adds protocols"""

  def addProtocolMultiview(session, protocol, group, purpose, session_id, recording_id, camera_id, shot_id):
    """Add a multiview protocol entry"""
    p = Protocol(protocol, group, purpose, session_id, recording_id, 'multiview')
    session.add(p)
    session.flush()
    session.refresh(p)
    session.add(ProtocolMultiview(p.id, camera_id, shot_id))

  # TODO: world subset

  # Protocol M: Enrol: No flash; Probe: No flash
  session.add(ProtocolName('M'))

  addProtocolMultiview(session, 'M', 'dev', 'enrol', 1, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'dev', 'enrol', 1, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 2, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'dev', 'probe', 2, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 3, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'dev', 'probe', 3, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 4, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'dev', 'probe', 4, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 4, 2, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'dev', 'probe', 4, 2, '05_1', 19)

  addProtocolMultiview(session, 'M', 'eval', 'enrol', 1, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'eval', 'enrol', 1, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 2, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'eval', 'probe', 2, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 3, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'eval', 'probe', 3, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 4, 1, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'eval', 'probe', 4, 1, '05_1', 19)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 4, 2, '05_1', 0)
  #addProtocolMultiview(session, 'M', 'eval', 'probe', 4, 2, '05_1', 19)

  # Protocol U: Enrol: No flash; Probe: No flash + Any flash
  session.add(ProtocolName('U'))

  addProtocolMultiview(session, 'U', 'dev', 'enrol', 1, 1, '05_1', 0)
  #addProtocolMultiview(session, 'U', 'dev', 'enrol', 1, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 0)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 1)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 2)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 3)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 4)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 5)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 6)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 7)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 8)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 9)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 10)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 11)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 12)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 13)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 14)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 15)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 16)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 17)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 0)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 1)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 2)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 3)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 4)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 5)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 6)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 7)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 8)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 9)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 10)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 11)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 12)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 13)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 14)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 15)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 16)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 17)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 0)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 1)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 2)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 3)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 4)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 5)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 6)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 7)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 8)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 9)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 10)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 11)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 12)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 13)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 14)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 15)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 16)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 17)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 0)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 1)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 2)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 3)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 4)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 5)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 6)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 7)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 8)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 9)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 10)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 11)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 12)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 13)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 14)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 15)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 16)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 17)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, '05_1', 19)

  addProtocolMultiview(session, 'U', 'eval', 'enrol', 1, 1, '05_1', 0)
  #addProtocolMultiview(session, 'U', 'eval', 'enrol', 1, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 0)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 1)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 2)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 3)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 4)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 5)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 6)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 7)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 8)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 9)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 10)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 11)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 12)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 13)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 14)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 15)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 16)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 17)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 0)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 1)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 2)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 3)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 4)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 5)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 6)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 7)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 8)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 9)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 10)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 11)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 12)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 13)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 14)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 15)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 16)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 17)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 0)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 1)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 2)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 3)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 4)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 5)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 6)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 7)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 8)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 9)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 10)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 11)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 12)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 13)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 14)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 15)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 16)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 17)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, '05_1', 19)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 0)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 1)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 2)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 3)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 4)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 5)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 6)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 7)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 8)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 9)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 10)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 11)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 12)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 13)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 14)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 15)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 16)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 17)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 18)
  #addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, '05_1', 19)

  # Protocol G: Enrol: No flash + Any flash; Probe: No flash + Any flash
  session.add(ProtocolName('G'))

  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 0)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 1)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 2)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 3)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 4)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 5)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 6)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 7)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 8)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 9)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 10)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 11)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 12)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 13)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 14)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 15)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 16)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 17)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, '05_1', 19)

  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 0)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 1)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 2)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 3)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 4)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 5)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 6)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 7)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 8)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 9)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 10)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 11)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 12)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 13)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 14)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 15)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 16)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 17)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, '05_1', 19)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 0)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 1)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 2)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 3)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 4)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 5)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 6)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 7)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 8)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 9)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 10)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 11)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 12)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 13)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 14)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 15)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 16)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 17)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 18)
  #addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, '05_1', 19)


def add_expressions(session):
  """Adds expressions"""

  session.add(Expression('neutral', 'multiview', 1, 1))
  session.add(Expression('smile', 'multiview', 1, 2))
  session.add(Expression('neutral', 'multiview', 2, 1))
  session.add(Expression('surprise', 'multiview', 2, 2))
  session.add(Expression('squint', 'multiview', 2, 3))
  session.add(Expression('neutral', 'multiview', 3, 1))
  session.add(Expression('smile', 'multiview', 3, 2))
  session.add(Expression('disgust', 'multiview', 3, 3))
  session.add(Expression('neutral', 'multiview', 4, 1))
  session.add(Expression('neutral', 'multiview', 4, 2))
  session.add(Expression('scream', 'multiview', 4, 3))
  session.add(Expression('neutral', 'highres', 1, 1))
  session.add(Expression('smile', 'highres', 1, 2))
  session.add(Expression('neutral', 'highres', 2, 1))
  session.add(Expression('neutral', 'highres', 3, 1))
  session.add(Expression('neutral', 'highres', 4, 1))

 
def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  File.metadata.create_all(engine)
  FileMultiview.metadata.create_all(engine)
  Client.metadata.create_all(engine)
  Expression.metadata.create_all(engine)
  ProtocolName.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)
  ProtocolMultiview.metadata.create_all(engine)

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
  add_clients(s, args.subjectlist)
  add_files(s, args.imagedir, args.all_poses)
  add_protocols(s)
  add_expressions(s)
  s.commit()
  s.close()

def add_command(subparsers):
  """Add specific subcommands that the action "create" can use"""

  parser = subparsers.add_parser('create', help=create.__doc__)

  parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  parser.add_argument('--imagedir', action='store', metavar='DIR',
      default='/idiap/resource/database/Multi-Pie/data',
      help="Change the relative path to the directory containing the images of the Multi-PIE database (defaults to %(default)s)")
  parser.add_argument('--subjectlist', action='store',
      default='/idiap/resource/database/Multi-Pie/meta/subject_list.txt',
      help="Change the file containing the subject list of the Multi-PIE database (defaults to %(default)s)")
  parser.add_argument('--all_poses', action='store_true', default=False,
      help="If not set, it will create the database for frontal faces only")
  
  parser.set_defaults(func=create) #action
