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

def add_subworldclients(session):
  """Adds subworld clients"""
  # 41
  l41 =  [ 21,  26,  31,  39,  66,  75,  81,  90,  98, 109, 114, 148, 152, 158, 165, 171, 174, 179, 182, 197, 
          207, 215, 226, 239, 244, 256, 271, 277, 303, 309, 310, 315, 321, 326, 327, 333, 336, 338, 339, 341, 
          342]
  for x in l41: session.add(SubworldClient('sub41', x))
  # 81
  l81 =  [ 16,  21,  26,  31,  37,  39,  51,  65,  66,  73,  75,  81,  84,  86,  87,  90,  94,  95,  98,  99,
          109, 114, 134, 142, 144, 148, 151, 152, 158, 164, 165, 171, 173, 174, 179, 182, 195, 197, 207, 215, 
          217, 222, 226, 239, 244, 247, 249, 251, 256, 259, 260, 263, 264, 265, 271, 272, 276, 277, 287, 298, 
          303, 304, 306, 309, 310, 312, 315, 317, 319, 320, 321, 324, 326, 327, 329, 333, 336, 338, 339, 341, 
          342]
  for x in l81: session.add(SubworldClient('sub81', x)) 
  # 121
  l121 = [  7,  16,  21,  24,  26,  30,  31,  37,  39,  51,  60,  61,  63,  65,  66,  72,  73,  75,  81,  84, 
           86,  87,  90,  91,  94,  95,  96,  98,  99, 109, 114, 134, 135, 142, 144, 148, 151, 152, 158, 159, 
          164, 165, 166, 171, 173, 174, 176, 179, 180, 182, 195, 197, 207, 210, 214, 215, 217, 221, 222, 226, 
          228, 231, 233, 234, 239, 242, 244, 247, 249, 251, 253, 254, 255, 256, 259, 260, 263, 264, 265, 268,
          271, 272, 276, 277, 278, 279, 285, 287, 291, 292, 293, 294, 297, 298, 300, 301, 303, 304, 306, 309, 
          310, 311, 312, 315, 317, 319, 320, 321, 322, 324, 325, 326, 327, 329, 333, 336, 338, 339, 341, 342, 
          344]
  for x in l121: session.add(SubworldClient('sub121', x)) 
  # 161
  l161 = [  7,  12,  13,  16,  21,  24,  26,  30,  31,  37,  39,  45,  51,  60,  61,  63,  65,  66,  72,  73, 
           75,  77,  81,  82,  84,  86,  87,  88,  90,  91,  93,  94,  95,  96,  98,  99, 101, 109, 114, 119,
          120, 121, 134, 135, 136, 142, 144, 148, 151, 152, 153, 158, 159, 160, 162, 163, 164, 165, 166, 171, 
          173, 174, 176, 179, 180, 182, 187, 195, 197, 200, 207, 210, 214, 215, 216, 217, 218, 219, 221, 222, 
          226, 228, 229, 231, 233, 234, 237, 239, 242, 244, 247, 249, 251, 253, 254, 255, 256, 257, 259, 260, 
          261, 263, 264, 265, 267, 268, 271, 272, 273, 276, 277, 278, 279, 285, 287, 289, 291, 292, 293, 294, 
          295, 296, 297, 298, 299, 300, 301, 303, 304, 306, 308, 309, 310, 311, 312, 313, 314, 315, 317, 319, 
          320, 321, 322, 323, 324, 325, 326, 327, 329, 333, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 
          346]
  for x in l161: session.add(SubworldClient('sub161', x)) 

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

  def addProtocolMultiview(session, protocol, group, purpose, session_id, recording_id, camera_shot_ids):
    """Add a multiview protocol entry"""
    p = Protocol(protocol, group, purpose, session_id, recording_id, 'multiview')
    session.add(p)
    session.flush()
    session.refresh(p)
    for camera_id, shot_id in camera_shot_ids:
      session.add(ProtocolMultiview(p.id, camera_id, shot_id))

  # TODO: world subset

  # ILLUMINATION (FRONTAL) PROTOCOLS
  cam_shot0 = [('05_1', 0)]
  cam_shot = [('05_1',  0), ('05_1',  1), ('05_1',  2), ('05_1',  3), ('05_1',  4), 
              ('05_1',  5), ('05_1',  6), ('05_1',  7), ('05_1',  8), ('05_1',  9), 
              ('05_1', 10), ('05_1', 11), ('05_1', 12), ('05_1', 13), ('05_1', 14), 
              ('05_1', 15), ('05_1', 16), ('05_1', 17), ('05_1', 18)] #('05_1', 19)

  # Protocol M: Enrol: No flash; Probe: No flash
  session.add(ProtocolName('M'))
  addProtocolMultiview(session, 'M', 'dev', 'enrol', 1, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 2, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 3, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 4, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'dev', 'probe', 4, 2, cam_shot0)

  addProtocolMultiview(session, 'M', 'eval', 'enrol', 1, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 2, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 3, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 4, 1, cam_shot0)
  addProtocolMultiview(session, 'M', 'eval', 'probe', 4, 2, cam_shot0)


  # Protocol U: Enrol: No flash; Probe: No flash + Any flash
  session.add(ProtocolName('U'))
  addProtocolMultiview(session, 'U', 'dev', 'enrol', 1, 1, cam_shot0)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 2, 1, cam_shot)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 3, 1, cam_shot)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 1, cam_shot)
  addProtocolMultiview(session, 'U', 'dev', 'probe', 4, 2, cam_shot)

  addProtocolMultiview(session, 'U', 'eval', 'enrol', 1, 1, cam_shot0)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 2, 1, cam_shot)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 3, 1, cam_shot)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 1, cam_shot)
  addProtocolMultiview(session, 'U', 'eval', 'probe', 4, 2, cam_shot)


  # Protocol G: Enrol: No flash + Any flash; Probe: No flash + Any flash
  session.add(ProtocolName('G'))
  addProtocolMultiview(session, 'G', 'dev', 'enrol', 1, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 2, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 3, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'dev', 'probe', 4, 2, cam_shot)

  addProtocolMultiview(session, 'G', 'eval', 'enrol', 1, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 2, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 3, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 1, cam_shot)
  addProtocolMultiview(session, 'G', 'eval', 'probe', 4, 2, cam_shot)


  # POSE PROTOCOLS
  cam240_shot = [('24_0', 0)]
  cam010_shot = [('01_0', 0)]
  cam200_shot = [('20_0', 0)]
  cam190_shot = [('19_0', 0)]
  cam041_shot = [('04_1', 0)]
 
  cam191_shot = [('19_1', 0)]
  cam050_shot = [('05_0', 0)]
  cam051_shot = [('05_1', 0)]
  cam140_shot = [('14_0', 0)]
  cam081_shot = [('08_1', 0)]

  cam130_shot = [('13_0', 0)]
  cam080_shot = [('08_0', 0)]
  cam090_shot = [('09_0', 0)]
  cam120_shot = [('12_0', 0)]
  cam110_shot = [('11_0', 0)]

  # Protocol P051: Enrol: 05_1; Probe: 05_1 (FRONTAL, SAME as 'M")
  session.add(ProtocolName('P051'))
  addProtocolMultiview(session, 'P051', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'dev', 'probe', 2, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'dev', 'probe', 3, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'dev', 'probe', 4, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'dev', 'probe', 4, 2, cam051_shot)

  addProtocolMultiview(session, 'P051', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'eval', 'probe', 2, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'eval', 'probe', 3, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'eval', 'probe', 4, 1, cam051_shot)
  addProtocolMultiview(session, 'P051', 'eval', 'probe', 4, 2, cam051_shot)

  # Protocol P050: Enrol: 05_1; Probe: 05_0
  session.add(ProtocolName('P050'))
  addProtocolMultiview(session, 'P050', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P050', 'dev', 'probe', 2, 1, cam050_shot)
  addProtocolMultiview(session, 'P050', 'dev', 'probe', 3, 1, cam050_shot)
  addProtocolMultiview(session, 'P050', 'dev', 'probe', 4, 1, cam050_shot)
  addProtocolMultiview(session, 'P050', 'dev', 'probe', 4, 2, cam050_shot)

  addProtocolMultiview(session, 'P050', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P050', 'eval', 'probe', 2, 1, cam050_shot)
  addProtocolMultiview(session, 'P050', 'eval', 'probe', 3, 1, cam050_shot)
  addProtocolMultiview(session, 'P050', 'eval', 'probe', 4, 1, cam050_shot)
  addProtocolMultiview(session, 'P050', 'eval', 'probe', 4, 2, cam050_shot)

  # Protocol P140: Enrol: 05_1; Probe: 14_0
  session.add(ProtocolName('P140'))
  addProtocolMultiview(session, 'P140', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P140', 'dev', 'probe', 2, 1, cam140_shot)
  addProtocolMultiview(session, 'P140', 'dev', 'probe', 3, 1, cam140_shot)
  addProtocolMultiview(session, 'P140', 'dev', 'probe', 4, 1, cam140_shot)
  addProtocolMultiview(session, 'P140', 'dev', 'probe', 4, 2, cam140_shot)

  addProtocolMultiview(session, 'P140', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P140', 'eval', 'probe', 2, 1, cam140_shot)
  addProtocolMultiview(session, 'P140', 'eval', 'probe', 3, 1, cam140_shot)
  addProtocolMultiview(session, 'P140', 'eval', 'probe', 4, 1, cam140_shot)
  addProtocolMultiview(session, 'P140', 'eval', 'probe', 4, 2, cam140_shot)

  # Protocol P041: Enrol: 05_1; Probe: 04_1
  session.add(ProtocolName('P041'))
  addProtocolMultiview(session, 'P041', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P041', 'dev', 'probe', 2, 1, cam041_shot)
  addProtocolMultiview(session, 'P041', 'dev', 'probe', 3, 1, cam041_shot)
  addProtocolMultiview(session, 'P041', 'dev', 'probe', 4, 1, cam041_shot)
  addProtocolMultiview(session, 'P041', 'dev', 'probe', 4, 2, cam041_shot)

  addProtocolMultiview(session, 'P041', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P041', 'eval', 'probe', 2, 1, cam041_shot)
  addProtocolMultiview(session, 'P041', 'eval', 'probe', 3, 1, cam041_shot)
  addProtocolMultiview(session, 'P041', 'eval', 'probe', 4, 1, cam041_shot)
  addProtocolMultiview(session, 'P041', 'eval', 'probe', 4, 2, cam041_shot)

  # Protocol P130: Enrol: 05_1; Probe: 13_0
  session.add(ProtocolName('P130'))
  addProtocolMultiview(session, 'P130', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P130', 'dev', 'probe', 2, 1, cam130_shot)
  addProtocolMultiview(session, 'P130', 'dev', 'probe', 3, 1, cam130_shot)
  addProtocolMultiview(session, 'P130', 'dev', 'probe', 4, 1, cam130_shot)
  addProtocolMultiview(session, 'P130', 'dev', 'probe', 4, 2, cam130_shot)

  addProtocolMultiview(session, 'P130', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P130', 'eval', 'probe', 2, 1, cam130_shot)
  addProtocolMultiview(session, 'P130', 'eval', 'probe', 3, 1, cam130_shot)
  addProtocolMultiview(session, 'P130', 'eval', 'probe', 4, 1, cam130_shot)
  addProtocolMultiview(session, 'P130', 'eval', 'probe', 4, 2, cam130_shot)


  # Protocol P110 (left profile): Enrol: 05_1; Probe: 11_0
  session.add(ProtocolName('P110'))
  addProtocolMultiview(session, 'P110', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P110', 'dev', 'probe', 2, 1, cam110_shot)
  addProtocolMultiview(session, 'P110', 'dev', 'probe', 3, 1, cam110_shot)
  addProtocolMultiview(session, 'P110', 'dev', 'probe', 4, 1, cam110_shot)
  addProtocolMultiview(session, 'P110', 'dev', 'probe', 4, 2, cam110_shot)

  addProtocolMultiview(session, 'P110', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P110', 'eval', 'probe', 2, 1, cam110_shot)
  addProtocolMultiview(session, 'P110', 'eval', 'probe', 3, 1, cam110_shot)
  addProtocolMultiview(session, 'P110', 'eval', 'probe', 4, 1, cam110_shot)
  addProtocolMultiview(session, 'P110', 'eval', 'probe', 4, 2, cam110_shot)

  # Protocol P240 (right profile): Enrol: 05_1; Probe: 24_0
  session.add(ProtocolName('P240'))
  addProtocolMultiview(session, 'P240', 'dev', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P240', 'dev', 'probe', 2, 1, cam240_shot)
  addProtocolMultiview(session, 'P240', 'dev', 'probe', 3, 1, cam240_shot)
  addProtocolMultiview(session, 'P240', 'dev', 'probe', 4, 1, cam240_shot)
  addProtocolMultiview(session, 'P240', 'dev', 'probe', 4, 2, cam240_shot)

  addProtocolMultiview(session, 'P240', 'eval', 'enrol', 1, 1, cam051_shot)
  addProtocolMultiview(session, 'P240', 'eval', 'probe', 2, 1, cam240_shot)
  addProtocolMultiview(session, 'P240', 'eval', 'probe', 3, 1, cam240_shot)
  addProtocolMultiview(session, 'P240', 'eval', 'probe', 4, 1, cam240_shot)
  addProtocolMultiview(session, 'P240', 'eval', 'probe', 4, 2, cam240_shot)


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

def add_fileprotocol(session):
  """Adds FileProcotol entries"""
  groups_de = ('dev','eval')
  purposes = ('enrol', 'probe')
  q = session.query(File.id, File.path, File.client_id, ProtocolName.name, Protocol.id, Protocol.purpose, ProtocolMultiview.id).\
        join(FileMultiview).join(Client).\
        filter(Client.sgroup.in_(groups_de)).\
        filter(and_(ProtocolName.name == Protocol.name, Protocol.sgroup == Client.sgroup,
                    Protocol.img_type == 'multiview',
                    Protocol.session_id == File.session_id, Protocol.recording_id == File.recording_id,
                    Protocol.purpose.in_(purposes))).\
        filter(and_(Protocol.id == ProtocolMultiview.protocol_id,
                    ProtocolMultiview.camera_id == FileMultiview.camera_id,
                    ProtocolMultiview.shot_id == FileMultiview.shot_id)).\
        order_by(ProtocolName.name, File.client_id, File.session_id, File.recording_id, FileMultiview.camera_id, FileMultiview.shot_id)
  for k in q:
    session.add(FileProtocol(k[0], k.name, k.purpose))

  protocols_I = ('M', 'U', 'G')
  q = session.query(File.id, File.path, File.client_id, Expression.name).join(FileMultiview).join(Client).\
        filter(Client.sgroup == 'world').\
        filter(Expression.name == 'neutral').\
        filter(and_(File.img_type == 'multiview', FileMultiview.camera_id == '05_1', File.session_id == Expression.session_id,
                    File.recording_id == Expression.recording_id, Expression.img_type == 'multiview', FileMultiview.shot_id != 19)).\
        order_by(File.client_id, File.session_id, File.recording_id, FileMultiview.camera_id, FileMultiview.shot_id)
  for k in q:
    for prot in protocols_I:
      session.add(FileProtocol(k[0], prot, 'world'))
  
  protocols_P = ('P051', 'P050', 'P140', 'P041', 'P130', 'P110', 'P240')
  cameras_P = ('05_1', '05_0', '14_0', '04_1', '13_0', '11_0', '24_0')
  for prot in range(len(protocols_P)):
    cams = []
    if prot == 0: cams = ['05_1']
    else: cams = ['05_1', cameras_P[prot]]
    q = session.query(File.id, File.path, File.client_id, Expression.name).join(FileMultiview).join(Client).\
          filter(Client.sgroup == 'world').\
          filter(Expression.name == 'neutral').\
          filter(and_(File.img_type == 'multiview', FileMultiview.camera_id.in_(cams), File.session_id == Expression.session_id,
                      File.recording_id == Expression.recording_id, Expression.img_type == 'multiview', FileMultiview.shot_id == 0)).\
          order_by(File.client_id, File.session_id, File.recording_id, FileMultiview.camera_id, FileMultiview.shot_id)
    for k in q:
      session.add(FileProtocol(k[0], protocols_P[prot], 'world'))
 
def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  File.metadata.create_all(engine)
  FileMultiview.metadata.create_all(engine)
  Client.metadata.create_all(engine)
  SubworldClient.metadata.create_all(engine)
  Expression.metadata.create_all(engine)
  ProtocolName.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)
  ProtocolMultiview.metadata.create_all(engine)
  FileProtocol.metadata.create_all(engine)

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
  add_subworldclients(s)
  add_files(s, args.imagedir, args.all_poses)
  add_protocols(s)
  add_expressions(s)
  add_fileprotocol(s)
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
