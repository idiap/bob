#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the XM2VTS database.
"""

import sqlalchemy
from sqlalchemy import Column, Integer, String, ForeignKey, or_, and_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'

  id = Column(Integer, primary_key=True)
  sgroup = Column(Enum('client','impostorDev','impostorEval')) # do NOT use group (SQL keyword)

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
  session_id = Column(Integer)
  shot_id = Column(Integer)

  # for Python
  client = relationship("Client", backref="client_file")

  def __init__(self, client_id, path, session_id, shot_id):
    self.client_id = client_id
    self.path = path
    self.session_id = session_id
    self.shot_id = shot_id

  def __repr__(self):
    print "<File('%s')>" % self.path


class Protocol(Base):
  __tablename__ = 'protocol'
  
  id = Column(Integer, primary_key=True)
  name = Column(Enum('lp1', 'lp2'))
  sgroup = Column(Enum('', 'dev', 'eval'))
  purpose = Column(Enum('enrol', 'probe'))
  session_id = Column(Integer)
  shot_id = Column(Integer)

  def __init__(self, name, group, purpose, session_id, shot_id):
    self.name = name
    self.sgroup = group
    self.purpose = purpose
    self.session_id = session_id
    self.shot_id = shot_id

  def __repr__(self):
    return "<Protocol('%d', '%s', '%s', '%s', '%d', '%d')>" % (self.session_id, self.name, self.sgroup, self.purpose, self.session_id, self.shot_id)
