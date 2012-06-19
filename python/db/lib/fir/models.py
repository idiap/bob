#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Salim Kayal <salim.kayal@idiap.ch>

"""Table models and functionality for the FIR database.
"""

import sqlalchemy
from sqlalchemy import Column, Integer, Boolean, String, ForeignKey, or_, and_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'

  id = Column(Integer, primary_key=True)
  sgroup = Column(Enum('world', 'dev', 'eval'))

  def __init__(self, id, group):
    self.id = id
    self.sgroup = group

  def __repr__(self):
    return "<Client('%d', '%s')>" % (self.id, self.sgroup)

class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  ir = Column(Boolean)
  location_id = Column(Integer)
  illumination_id = Column(Integer)
  shot_id = Column(Integer)

  # for Python
  client = relationship("Client", backref="client_file")

  def __init__(self, client_id, path, ir, location_id, illumination_id, shot_id):
    self.client_id = client_id
    self.path = path
    self.ir = ir
    self.location_id = location_id
    self.illumination_id = illumination_id
    self.shot_id = shot_id

  def __repr__(self):
    print "<File('%s')>" % self.path


class Protocol(Base):
  __tablename__ = 'protocol'
  
  name = Column(Enum('ir', 'noir'))
  sgroup = Column(Enum('world', 'dev', 'eval'), primary_key=True)
  purpose = Column(Enum('enrol', 'probe'))
  ir = Column(Boolean, primary_key=True)
  location_id = Column(Integer, primary_key=True)
  illumination_id = Column(Integer, primary_key=True)
  shot_id = Column(Integer, primary_key=True)

  def __init__(self, name, group, purpose, ir, location_id, illumination_id, shot_id):
    self.name = name
    self.sgroup = group
    self.purpose = purpose
    self.ir = ir
    self.location_id = location_id
    self.illumination_id = illumination_id
    self.shot_id = shot_id

  def __repr__(self):
    return "<Protocol('%s', '%s', '%s', '%s', '%d', '%d', '%d')>" % (self.name, self.sgroup, self.purpose, self.ir, self.location_id, self.illumination_id, self.shot_id)
