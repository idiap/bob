#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the SCFace database.
"""

from sqlalchemy import Column, Integer, Boolean, String, ForeignKey, or_, and_, not_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'
  
  id = Column(Integer, primary_key=True)
  sgroup = Column(Enum('dev','eval','world')) # do NOT use group (SQL keyword)
  birthyear = Column(Integer)
  gender = Column(Enum('m','f'))
  beard = Column(Boolean)
  moustache = Column(Boolean)
  glasses = Column(Boolean)

  def __init__(self, id, group, birthyear, gender, beard, moustache, glasses):
    self.id = id
    self.sgroup = group
    self.birthyear = birthyear
    self.gender = gender
    self.beard = beard
    self.moustache = moustache
    self.glasses = glasses

  def __repr__(self):
    return "<Client('%d', '%s', '%d', '%s', '%d', '%d', '%d')>" % (self.id, self.sgroup, self.birthyear, self.gender, self.beard, self.moustache, self.glasses)

class Subworld(Base):
  __tablename__ = 'subworld'
  
  id = Column(Integer, primary_key=True)
  name = Column(Enum('onethird','twothirds'))
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  
  # for Python
  real_client = relationship("Client", backref=backref("client_subworld"))

  def __init__(self, name, client_id):
    self.name = name
    self.client_id = client_id

  def __repr__(self):
    print "<Subworld('%s', '%d')>" % (self.name, self.client_id)

class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  camera = Column(String(8))
  distance = Column(Integer)

  # for Python
  client = relationship("Client", backref=backref("client_file"))
 
  def __init__(self, client_id, path, camera, distance):
    self.client_id = client_id
    self.path = path
    self.camera = camera
    self.distance = distance

  def __repr__(self):
    print "<File('%s')>" % self.path

class Protocol(Base):
  __tablename__ = 'protocol'
  
  id = Column(Integer, primary_key=True)
  name = Column(String(10))
  purpose = Column(Enum('enrol', 'probe'))
  camera = Column(String(8))
  distance = Column(Integer)

  def __init__(self, name, purpose, camera, distance):
    self.name = name
    self.purpose = purpose
    self.camera = camera
    self.distance = distance

  def __repr__(self):
    return "<Protocol('%s', '%s', '%s', '%d')>" % (self.name, self.purpose, self.camera, self.distance)

