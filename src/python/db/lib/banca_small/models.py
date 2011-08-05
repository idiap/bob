#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the BANCA_SMALL database.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, or_, and_
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'

  id = Column(Integer, primary_key=True)
  gender = Column(Enum('m','f'))
  sgroup = Column(Enum('g1','world')) # do NOT use group (SQL keyword)

  def __init__(self, id, gender, group, language):
    self.id = id
    self.gender = gender
    self.sgroup = group

  def __repr__(self):
    return "<Client('%d', '%s', '%s')>" % (self.id, self.gender, self.sgroup)

class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  real_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  claimed_id = Column(Integer) # not always the id of an existing client model
  shot = Column(Integer)
  session_id = Column(Integer)

  # for Python
  real_client = relationship("Client", backref=backref("real_client_file"))

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
  
  id = Column(Integer, primary_key=True)
  session_id = Column(Integer)
  name = Column(Enum('P'))
  purpose = Column(Enum('enrol','probe','probeImpostor'))

  def __init__(self, session_id, name, purpose):
    self.session_id = session_id
    self.name = name
    self.purpose = purpose

  def __repr__(self):
    return "<Protocol('%d', '%s', '%s')>" % (self.session_id, self.name, self.purpose)
