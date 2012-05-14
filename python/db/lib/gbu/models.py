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

"""Table models and functionality for the GBU database.
"""

import os
from sqlalchemy import Column, Integer, String, ForeignKey, or_, and_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'
  
  m_signature = Column(String(9), primary_key=True) # The client id; should start with nd1
  
  def __init__(self, signature):
    self.m_signature = signature
    
  def __repr__(self):
    return "<Client('%s')>" % self.m_signature


class File(Base):
  __tablename__ = 'file'

  m_presentation = Column(String(9), primary_key=True) # The id of the file; should start with nd2
  m_signature = Column(String(9), ForeignKey('client.m_signature')) # The client id; should start with nd1
  m_directory = Column(String(100)) # The relative directory where to find the file
#  m_filename = Column(String(9), unique=True) # The name of the file; usually starts with 0
  m_filename = Column(String(9)) # The name of the file; usually starts with 0
  m_le_x = Column(Integer) # left eye 
  m_le_y = Column(Integer)
  m_re_x = Column(Integer) # right eye 
  m_re_y = Column(Integer) 

  # for Python

  def __init__(self, presentation = None, signature = None, filename = None, directory = None):
    self.m_presentation = presentation
    self.m_signature = signature
    self.m_filename = filename
    self.m_directory = directory
    self.m_le_x = self.m_le_y = self.m_re_x = self.m_re_y = None
    
  def eyes(self, eyes):
    """Set the eye positions"""
    assert len(eyes) == 4
    self.m_re_x = int(eyes[0])
    self.m_re_y = int(eyes[1])
    self.m_le_x = int(eyes[2])
    self.m_le_y = int(eyes[3])

  def __repr__(self):
    return "<File('%s, %s')>" % (os.path.join(self.m_directory, self.m_filename), self.m_presentation)

class Trainset(Base):
  __tablename__ = 'world'

  m_names = ('x1', 'x2', 'x4', 'x8')
  
  m_id = Column(Integer, primary_key=True)
  m_name = Column(Enum(*m_names))
  
  m_file = Column(String(9), ForeignKey('file.m_presentation'))

  def __init__(self, name, presentation):
    self.m_name = name
    self.m_file = presentation

  def __repr__(self):
    return "<Trainset('%s', %s)>" % (self.m_name, self.m_file)
  

class Protocol(Base):
  __tablename__ = 'protocol'

  m_types = ('gbu', 'multi')
  m_names = ('Good', 'Bad', 'Ugly')
  m_purposes = ('enrol', 'probe')
  
  m_id = Column(Integer, primary_key=True)
  m_type = Column(Enum(*m_types))  # Use the default GBU protocol (one file per model enrollment) or Idiap's (multiple file for enrollment)
  m_name = Column(Enum(*m_names)) # Different training and test protocols
  m_purpose = Column(Enum(*m_purposes))
  
  m_file = Column(String(9), ForeignKey('file.m_presentation'))

  def __init__(self, type, name, purpose, presentation):
    self.m_type = type
    self.m_name = name
    self.m_purpose = purpose
    self.m_file = presentation

  def __repr__(self):
    return "<Protocol('%s', '%s', %s, %s)>" % (self.m_type, self.m_name, self.m_purpose, self.m_file)
