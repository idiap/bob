#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Thu May 24 10:41:42 CEST 2012
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

"""Table models and functionality for the LFW database.
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
  
  m_name = Column(String(100), primary_key=True)

  def __init__(self, name):
    self.m_name = name

  def __repr__(self):
    return "<Client('%s')>" % self.m_name

class File(Base):
  """Information about the files of the LFW database"""
  __tablename__ = 'file'

  m_id = Column(String(100), primary_key=True)
  m_client_id = Column(String(100), ForeignKey('client.m_name'))
  m_shot_id = Column(Integer)
  m_path = Column(String(100))

  def __init__(self, client_id, shot_id):
    self.m_client_id = client_id
    self.m_shot_id = shot_id
    self.m_id = client_id + "_" + "0"*(4-len(str(shot_id))) + str(shot_id)
    self.m_path = os.path.join(client_id, self.m_id)

  def __repr__(self):
    print "<File('%s')>" % os.path.join(self.m_client_id, self.m_id)

class People(Base):
  """Information about the people (as given in the people.txt file) of the LFW database"""
  __tablename__ = 'people'
  
  m_id = Column(Integer, primary_key=True)
  m_protocol = Column(Enum('train', 'test', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10'))
  m_file_id = Column(String(100), ForeignKey('file.m_id'))

  def __init__(self, protocol, file_id):
    self.m_protocol = protocol
    self.m_file_id = file_id
    
  def __repr__(self):
    return "<People('%s', '%s')>" % (self.m_protocol, self.m_file_id)

class Pair(Base):
  """Information of the pairs (as given in the pairs.txt files) of the LFW database"""
  __tablename__ = 'pair'

  m_id = Column(Integer, primary_key=True)
  # train and test for view1, the folds for view2
  m_protocol = Column(Enum('train', 'test', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10'))
  m_enrol_file = Column(String(100), ForeignKey('file.m_id'))
  m_probe_file = Column(String(100), ForeignKey('file.m_id'))
  m_is_match = Column(Boolean)

  def __init__(self, protocol, enrol_file, probe_file, is_match):
    self.m_protocol = protocol
    self.m_enrol_file = enrol_file
    self.m_probe_file = probe_file
    self.m_is_match = is_match

  def __repr__(self):
    return "<Pair('%s', '%s', '%s', '%d')>" % (self.protocol, self.m_enrol_file, self.m_probe_file, 1 if self.m_is_match else 0)

