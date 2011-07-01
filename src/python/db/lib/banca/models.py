#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the BANCA database.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, or_, and_
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'

  from sqlalchemy import Enum # import locally (not supported by old 10.04 Ubuntu package)
  id = Column(Integer, primary_key=True)
  gender = Column(Enum('m','f'))
  sgroup = Column(Enum('g1','g2','world')) # do NOT use group (SQL keyword)
  language = Column(Enum('en','fr','sp'))

  def __init__(self, id, gender, group, language):
    self.id = id
    self.gender = gender
    self.sgroup = group
    self.language = language

  def __repr__(self):
    return "<Client('%d', '%s', '%s', '%s')>" % (self.id, self.gender, self.sgroup, self.language)


class Session(Base):
  __tablename__ = 'session'
  
  from sqlalchemy import Enum # import locally (not supported by old 10.04 Ubuntu package)
  id = Column(Integer, primary_key=True)
  scenario = Column(Enum('controlled','degraded','adverse'))

  def __init__(self, id, scenario):
    self.id = id
    self.scenario = scenario

  def __repr__(self):
    return "<Session('%d', '%s')>" % (self.id, self.scenario)


class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  real_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  claimed_id = Column(Integer) # not always the id of an existing client model
  shot = Column(Integer)
  session_id = Column(Integer, ForeignKey('session.id'))

  # for Python
  session = relationship("Session", backref="session_file")
  real_client = relationship("Client", backref="real_client_file")

  def __init__(self, real_id, path, claimed_id, shot, session_id):
    self.real_id = real_id
    self.path = path
    self.claimed_id = claimed_id
    self.shot = shot
    self.session_id = session_id

  def __repr__(self):
    print "<File('%s')>" % self.path


class Protocol(Base):
  __tablename__ = 'protocol'
  
  from sqlalchemy import Enum # import locally (not supported by old 10.04 Ubuntu package)
  id = Column(Integer, primary_key=True)
  session_id = Column(Integer, ForeignKey('session.id'))
  name = Column(Enum('P','G','Mc','Md','Ma','Ud','Ua'))
  purpose = Column(Enum('enrol','probe','probeImpostor'))

  # for python
  session = relationship("Session", backref="session_protocol")

  def __init__(self, session_id, name, purpose):
    self.session_id = session_id
    self.name = name
    self.purpose = purpose

  def __repr__(self):
    return "<Protocol('%d', '%s', '%s')>" % (self.session_id, self.name, self.purpose)
