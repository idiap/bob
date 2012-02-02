#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This script creates the Biosecure database in a single pass.
"""

import os

from .models import *
from ..utils import session, check_group_writeability


def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_clients(session):
  """Add clients to the Biosecure database."""

  # world
  w = 'world'
  session.add(Client(3, w))
  session.add(Client(5, w))
  session.add(Client(10, w))
  session.add(Client(12, w))
  session.add(Client(13, w))
  session.add(Client(14, w))
  session.add(Client(15, w))
  session.add(Client(17, w))
  session.add(Client(23, w))
  session.add(Client(24, w))
  session.add(Client(29, w))
  session.add(Client(33, w))
  session.add(Client(36, w))
  session.add(Client(37, w))
  session.add(Client(38, w))
  session.add(Client(39, w))
  session.add(Client(42, w))
  session.add(Client(47, w))
  session.add(Client(58, w))
  session.add(Client(62, w))
  session.add(Client(64, w))
  session.add(Client(73, w))
  session.add(Client(78, w))
  session.add(Client(80, w))
  session.add(Client(81, w))
  session.add(Client(82, w))
  session.add(Client(83, w))
  session.add(Client(85, w))
  session.add(Client(89, w))
  session.add(Client(93, w))
  session.add(Client(102, w))
  session.add(Client(104, w))
  session.add(Client(107, w))
  session.add(Client(108, w))
  session.add(Client(109, w))
  session.add(Client(111, w))
  session.add(Client(112, w))
  session.add(Client(117, w))
  session.add(Client(119, w))
  session.add(Client(123, w))
  session.add(Client(126, w))
  session.add(Client(130, w))
  session.add(Client(133, w))
  session.add(Client(137, w))
  session.add(Client(138, w))
  session.add(Client(143, w))
  session.add(Client(146, w))
  session.add(Client(147, w))
  session.add(Client(150, w))
  session.add(Client(152, w))
  session.add(Client(154, w))
  session.add(Client(155, w))
  session.add(Client(156, w))
  session.add(Client(158, w))
  session.add(Client(160, w))
  session.add(Client(163, w))
  session.add(Client(164, w))
  session.add(Client(176, w))
  session.add(Client(178, w))
  session.add(Client(180, w))
  session.add(Client(183, w))
  session.add(Client(196, w))
  session.add(Client(198, w))
  session.add(Client(199, w))
  session.add(Client(200, w))
  session.add(Client(201, w))
  session.add(Client(203, w))
  session.add(Client(206, w))
  session.add(Client(209, w))
  session.add(Client(210, w))
  # dev
  d = 'dev'
  session.add(Client(6, d))
  session.add(Client(7, d))
  session.add(Client(16, d))
  session.add(Client(18, d))
  session.add(Client(19, d))
  session.add(Client(20, d))
  session.add(Client(21, d))
  session.add(Client(22, d))
  session.add(Client(25, d))
  session.add(Client(27, d))
  session.add(Client(28, d))
  session.add(Client(32, d))
  session.add(Client(40, d))
  session.add(Client(41, d))
  session.add(Client(49, d))
  session.add(Client(50, d))
  session.add(Client(52, d))
  session.add(Client(54, d))
  session.add(Client(55, d))
  session.add(Client(60, d))
  session.add(Client(63, d))
  session.add(Client(67, d))
  session.add(Client(68, d))
  session.add(Client(69, d))
  session.add(Client(70, d))
  session.add(Client(75, d))
  session.add(Client(76, d))
  session.add(Client(79, d))
  session.add(Client(84, d))
  session.add(Client(88, d))
  session.add(Client(92, d))
  session.add(Client(94, d))
  session.add(Client(96, d))
  session.add(Client(97, d))
  session.add(Client(98, d))
  session.add(Client(99, d))
  session.add(Client(103, d))
  session.add(Client(105, d))
  session.add(Client(115, d))
  session.add(Client(118, d))
  session.add(Client(120, d))
  session.add(Client(121, d))
  session.add(Client(122, d))
  session.add(Client(124, d))
  session.add(Client(127, d))
  session.add(Client(129, d))
  session.add(Client(131, d))
  session.add(Client(134, d))
  session.add(Client(135, d))
  session.add(Client(136, d))
  session.add(Client(141, d))
  session.add(Client(142, d))
  session.add(Client(145, d))
  session.add(Client(153, d))
  session.add(Client(157, d))
  session.add(Client(159, d))
  session.add(Client(165, d))
  session.add(Client(166, d))
  session.add(Client(168, d))
  session.add(Client(169, d))
  session.add(Client(170, d))
  session.add(Client(172, d))
  session.add(Client(175, d))
  session.add(Client(184, d))
  session.add(Client(185, d))
  session.add(Client(190, d))
  session.add(Client(193, d))
  session.add(Client(194, d))
  session.add(Client(204, d))
  session.add(Client(208, d))
  # eval
  e = 'eval'
  session.add(Client(1, e))
  session.add(Client(2, e))
  session.add(Client(4, e))
  session.add(Client(8, e))
  session.add(Client(9, e))
  session.add(Client(11, e))
  session.add(Client(26, e))
  session.add(Client(30, e))
  session.add(Client(31, e))
  session.add(Client(34, e))
  session.add(Client(35, e))
  session.add(Client(43, e))
  session.add(Client(44, e))
  session.add(Client(45, e))
  session.add(Client(46, e))
  session.add(Client(48, e))
  session.add(Client(51, e))
  session.add(Client(53, e))
  session.add(Client(56, e))
  session.add(Client(57, e))
  session.add(Client(59, e))
  session.add(Client(61, e))
  session.add(Client(65, e))
  session.add(Client(66, e))
  session.add(Client(71, e))
  session.add(Client(72, e))
  session.add(Client(74, e))
  session.add(Client(77, e))
  session.add(Client(86, e))
  session.add(Client(87, e))
  session.add(Client(90, e))
  session.add(Client(91, e))
  session.add(Client(95, e))
  session.add(Client(100, e))
  session.add(Client(101, e))
  session.add(Client(106, e))
  session.add(Client(110, e))
  session.add(Client(113, e))
  session.add(Client(114, e))
  session.add(Client(116, e))
  session.add(Client(125, e))
  session.add(Client(128, e))
  session.add(Client(132, e))
  session.add(Client(139, e))
  session.add(Client(140, e))
  session.add(Client(144, e))
  session.add(Client(148, e))
  session.add(Client(149, e))
  session.add(Client(151, e))
  session.add(Client(161, e))
  session.add(Client(162, e))
  session.add(Client(167, e))
  session.add(Client(171, e))
  session.add(Client(173, e))
  session.add(Client(174, e))
  session.add(Client(177, e))
  session.add(Client(179, e))
  session.add(Client(181, e))
  session.add(Client(182, e))
  session.add(Client(186, e))
  session.add(Client(187, e))
  session.add(Client(188, e))
  session.add(Client(189, e))
  session.add(Client(191, e))
  session.add(Client(192, e))
  session.add(Client(195, e))
  session.add(Client(197, e))
  session.add(Client(202, e))
  session.add(Client(205, e))
  session.add(Client(207, e))

def add_files(session, imagedir):
  """Add files to the Biosecure database."""
 
  def add_file(session, basename):
    """Parse a single filename and add it to the list.
       Also add a client entry if not already in the database."""
    v = os.path.splitext(basename)[0].split('_')
    cam = v[3][4:8]
    if cam[0:2] == 'ca':
      cam = cam[0:2] + cam[3]
    else:
      cam = cam[0:2]
    session.add(File(int(v[0][1:4]), os.path.join(cam, basename), int(v[1][2]), cam, int(v[4])))

  for camera in filter(nodot, os.listdir(imagedir)):
    if not camera in ['ca0', 'caf', 'wc']:
      continue

    camera_dir = os.path.join(imagedir, camera)
    for filename in filter(nodot, os.listdir(camera_dir)):
      basename, extension = os.path.splitext(filename)
      add_file(session, basename)
   

def add_protocols(session):
  """Adds protocols"""

  # Protocols (using a tuple allows to add mixed cameras protocols later on)
  session.add(Protocol('ca0', 'ca0'))
  session.add(Protocol('caf', 'caf'))
  session.add(Protocol('wc', 'wc'))

  # ProtocolPurposes
  session.add(ProtocolPurpose('ca0', 'world', 'world', 1))
  session.add(ProtocolPurpose('ca0', 'world', 'world', 2))
  session.add(ProtocolPurpose('ca0', 'dev', 'enrol', 1))
  session.add(ProtocolPurpose('ca0', 'dev', 'probe', 2))
  session.add(ProtocolPurpose('ca0', 'eval', 'enrol', 1))
  session.add(ProtocolPurpose('ca0', 'eval', 'probe', 2))
  session.add(ProtocolPurpose('caf', 'world', 'world', 1))
  session.add(ProtocolPurpose('caf', 'world', 'world', 2))
  session.add(ProtocolPurpose('caf', 'dev', 'enrol', 1))
  session.add(ProtocolPurpose('caf', 'dev', 'probe', 2))
  session.add(ProtocolPurpose('caf', 'eval', 'enrol', 1))
  session.add(ProtocolPurpose('caf', 'eval', 'probe', 2))
  session.add(ProtocolPurpose('wc', 'world', 'world', 1))
  session.add(ProtocolPurpose('wc', 'world', 'world', 2))
  session.add(ProtocolPurpose('wc', 'dev', 'enrol', 1))
  session.add(ProtocolPurpose('wc', 'dev', 'probe', 2))
  session.add(ProtocolPurpose('wc', 'eval', 'enrol', 1))
  session.add(ProtocolPurpose('wc', 'eval', 'probe', 2))


def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  File.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)
  ProtocolPurpose.metadata.create_all(engine)

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
  add_clients(s)
  add_files(s, args.imagedir)
  add_protocols(s)
  s.commit()
  s.close()

  # the group writeability option
  check_group_writeability(dbfile)

def add_command(subparsers):
  """Add specific subcommands that the action "create" can use"""

  parser = subparsers.add_parser('create', help=create.__doc__)

  parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  parser.add_argument('--imagedir', action='store', metavar='DIR',
      default='/idiap/temp/cmccool/databases/biosecure/raw',
      #default='/idiap/resource/database/biosecure',
      help="Change the relative path to the directory containing the images of the Biosecure database (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
