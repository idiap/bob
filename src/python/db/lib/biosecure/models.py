#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the Biosecure database.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, or_, and_, not_
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'
  
  from sqlalchemy import Enum # import locally (not supported by old 10.04 Ubuntu package)
  id = Column(Integer, primary_key=True)
  sgroup = Column(Enum('dev','eval','world')) # do NOT use group (SQL keyword)

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
  session = Column(Integer)
  camera = Column(String(4))
  shot = Column(Integer)

  # for Python
  client = relationship("Client", backref=backref("client_file"))
 
  def __init__(self, client_id, path, session, camera, shot):
    self.client_id = client_id
    self.path = path
    self.session = session
    self.camera = camera
    self.shot = shot

  def __repr__(self):
    print "<File('%s')>" % self.path

class Protocol(Base):
  __tablename__ = 'protocol'
  
  name = Column(String(4), primary_key=True)
  camera = Column(String(4))

  def __init__(self, name, camera):
    self.name = name
    self.camera = camera

  def __repr__(self):
    return "<Protocol('%s', '%s')>" % (self.name, self.camera)


class ProtocolPurpose(Base):
  __tablename__ = 'protocolPurpose'
  
  from sqlalchemy import Enum # import locally (not supported by old 10.04 Ubuntu package)
  id = Column(Integer, primary_key=True)
  name = Column(String(4), ForeignKey('protocol.name')) # for SQL
  sgroup = Column(Enum('dev','eval','world')) # DO NOT USE GROUP (LIKELY KEYWORD)
  purpose = Column(Enum('enrol', 'probe', 'world'))
  session = Column(Integer)

  # for Python
  protocol = relationship("Protocol", backref=backref("protocol_protocolPurpose"))

  def __init__(self, name, group, purpose, session):
    self.name = name
    self.sgroup = group
    self.purpose = purpose
    self.session = session

  def __repr__(self):
    return "<ProtocolPurpose('%s', '%s', '%s', '%d')>" % (self.name, self.sgroup, self.purpose, self.session)

