#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the LFW database.
"""

import sqlalchemy
from sqlalchemy import Column, Integer, String, ForeignKey, or_, and_, not_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'
  
  id = Column(Integer, primary_key=True)
  name = Column(String(100), unique=True)
  #sgroup = Column(Enum('dev','eval','world')) # do NOT use group (SQL keyword)

  def __init__(self, name):
    #self.sgroup = group
    self.name = name

  def __repr__(self):
    return "<Client('%d', '%s')>" % (self.id, self.name)

class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  shot_id = Column(Integer)

  # for Python
  client = relationship("Client", backref=backref("client_file"))
 
  def __init__(self, client_id, path, shot_id):
    self.client_id = client_id
    self.path = path
    self.shot_id = shot_id

  def __repr__(self):
    print "<File('%s')>" % self.path

class Pair(Base):
  __tablename__ = 'pair'

  id = Column(Integer, primary_key=True)
  view = Column(Enum('view1', 'view2'))
  subset = Column(Enum('train', 'test', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10'))
  client_id1 = Column(Integer, ForeignKey('client.id'))
  file_id1 = Column(Integer, ForeignKey('file.id'))
  client_id2 = Column(Integer, ForeignKey('client.id'))
  file_id2 = Column(Integer, ForeignKey('file.id'))

  def __init__(self, view, subset, client_id1, file_id1, client_id2, file_id2):
    self.view = view
    self.subset = subset
    self.client_id1 = client_id1
    self.file_id1 = file_id1
    self.client_id2 = client_id2
    self.file_id2 = file_id2

  def __repr__(self):
    return "<Pair('%s', '%s', '%d', '%d', '%d', '%d')>" % (self.view, self.subset, self.client_id1, self.file_id1, self.client_id2, self.file_id2)
