#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Wed Jul  4 14:12:51 CEST 2012
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

"""Table models and functionality for the AR face database.
"""

import sqlalchemy
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, or_, and_, not_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

import os

Base = declarative_base()

class Client(Base):
  """Information about the clients (identities) of the LFW database"""
  __tablename__ = 'client'
  
  # We define the possible values for the member variables as STATIC class variables
  s_genders = ('m', 'w')
  s_groups = ('world', 'dev', 'eval')
  
  m_id = Column(String(100), primary_key=True)
  m_gender = Column(Enum(*s_genders))
  m_group = Column(Enum(*s_groups))

  def __init__(self, id, group):
    self.m_id = id
    self.m_gender = id[0:1]
    self.m_group = group

  def __repr__(self):
    return "<Client('%s')>" % self.m_id

class File(Base):
  """Information about the files of the LFW database"""
  __tablename__ = 'file'
  
  # We define the possible values for the member variables as STATIC class variables
  s_sessions = ('first', 'second')
  s_purposes = ('enrol', 'probe')
  s_expressions = ('neutral', 'smile', 'anger', 'scream')
  s_illuminations = ('front', 'left', 'right', 'all')
  s_occlusions = ('none', 'sunglasses', 'scarf')

  m_id = Column(String(100), primary_key=True)
  m_client_id = Column(String(100), ForeignKey('client.m_id'))
  m_session = Column(Enum(*s_sessions))
  m_purpose = Column(Enum(*s_purposes))
  m_expression = Column(Enum(*s_expressions))
  m_illumination = Column(Enum(*s_illuminations))
  m_occlusion = Column(Enum(*s_occlusions))

  def __init__(self, image_name):
    self.m_id = image_name
    self.m_client_id = image_name[:5]
    
    # get shot id
    shot_id = int(os.path.splitext(image_name)[0][6:])
    # automatically fill member variables accorsing to shot id
    self.m_session = self.s_sessions[(shot_id-1) / 13]
    shot_id = (shot_id-1) % 13 + 1
    
    self.m_purpose = self.s_purposes[0 if shot_id == 1 else 1]
    
    self.m_expression = self.s_expressions[shot_id - 1] if shot_id in (2,3,4) else self.s_expressions[0]  
                        
    self.m_illumination = self.s_illuminations[shot_id - 4]  if shot_id in (5,6,7) else \
                          self.s_illuminations[shot_id - 8]  if shot_id in (9,10) else \
                          self.s_illuminations[shot_id - 11] if shot_id in (12,13) else \
                          self.s_illuminations[0]
                          
    self.m_occlusion = self.s_occlusions[1] if shot_id in (8,9,10) else \
                       self.s_occlusions[2] if shot_id in (11,12,13) else \
                       self.s_occlusions[0]
    

  def __repr__(self):
    print "<File('%s')>" % self.m_id


class Protocol(Base):
  """Information of the pairs (as given in the pairs.txt files) of the LFW database"""
  __tablename__ = 'pair'
  
  s_protocols = ('all', 'expression', 'illumination', 'occlusion', 'occlusion_and_illumination')

  m_id = Column(Integer, primary_key=True)
  m_protocol = Column(Enum(*s_protocols))
  m_session = Column(Enum(*File.s_sessions), ForeignKey('file.m_session'))
  m_expression = Column(Enum(*File.s_expressions), ForeignKey('file.m_expression'))
  m_illumination = Column(Enum(*File.s_illuminations), ForeignKey('file.m_illumination'))
  m_occlusion = Column(Enum(*File.s_occlusions), ForeignKey('file.m_occlusion'))
  

  def __init__(self, protocol, session, expression = 'neutral', illumination = 'front', occlusion = 'none'):
    self.m_protocol = protocol
    self.m_session = session
    self.m_expression = expression
    self.m_illumination = illumination
    self.m_occlusion = occlusion

  def __repr__(self):
    return "<Pair('%s', '%s', '%s', '%s', '%s')>" % (self.m_protocol, self.m_session, self.m_expression, self.m_illumination, self.m_occlusion)

