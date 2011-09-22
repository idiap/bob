#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Table models and functionality for the Mobio database.
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
  sgroup = Column(Enum('dev','eval','world')) # do NOT use group (SQL keyword)
  gender = Column(Enum('f','m'))
  institute = Column(Enum('idiap', 'manchester', 'surrey', 'oulu', 'brno', 'avignon'))

  def __init__(self, id, group, gender, institute):
    self.id = id
    self.sgroup = group
    self.gender = gender
    self.institute = institute

  def __repr__(self):
    return "<Client('%d', '%s', '%s', '%s')>" % (self.id, self.sgroup, self.gender, self.institute)

class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  session_id = Column(Integer)
  speech_type = Column(Enum('p','l','r','f'))
  shot_id = Column(Integer)
  environment = Column(Enum('i','o'))
  device = Column(Enum('mobile', 'laptop'))
  channel_id = Column(Integer)

  # for Python
  client = relationship("Client", backref=backref("client_file"))
 
  def __init__(self, client_id, path, session_id, speech_type, shot_id, environment, device, channel_id):
    self.client_id = client_id
    self.path = path
    self.session_id = session_id
    self.speech_type = speech_type
    self.shot_id = shot_id
    self.environment = environment
    self.device = device
    self.channel_id = channel_id

  def __repr__(self):
    print "<File('%s')>" % self.path

class Protocol(Base):
  __tablename__ = 'protocol'
  
  id = Column(Integer, primary_key=True)
  name = Column(String(8))
  purpose = Column(Enum('enrol', 'probe'))
  speech_type = Column(Enum('p','l','r','f'))
  
  def __init__(self, name, purpose, speech_type):
    self.name = name
    self.purpose = purpose
    self.speech_type = speech_type

  def __repr__(self):
    return "<Protocol('%s', '%s', '%s')>" % (self.name, self.purpose, self.speech_type)

class ProtocolEnrolSession(Base):
  __tablename__ = 'protocolEnrolSession'
  
  id = Column(Integer, primary_key=True)
  name = Column(String(8))
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  session_id = Column(Integer)
  
  # for Python
  client = relationship("Client", backref=backref("client_protocolEnrolSession"))
 
  def __init__(self, name, client_id, session_id):
    self.name = name
    self.client_id = client_id
    self.session_id = session_id

  def __repr__(self):
    return "<ProtocolEnrolSession('%s', '%d', '%d')>" % (self.name, self.client_id, self.session_id)

class TModel(Base):
  __tablename__ = 'tmodel'

  id = Column(String(9), primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  session_id = Column(Integer)
  speech_type = Column(Enum('p','l','r','f'))

  # for Python
  client = relationship("Client", backref=backref("client_tmodel"))
 
  def __init__(self, mid, client_id, session_id, speech_type):
    self.id = mid
    self.client_id = client_id
    self.session_id = session_id
    self.speech_type = speech_type

  def __repr__(self):
    return "<TModel('%d', '%d', '%s')>" % (self.client_id, self.session_id, self.speech_type) 

class ZClient(Base):
  __tablename__ = 'zclient'

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL

  # for Python
  client = relationship("Client", backref=backref("client_zfile"))
 
  def __init__(self, client_id):
    self.client_id = client_id

  def __repr__(self):
    return "<ZClient('%d')>" % (self.client_id) 

class SubworldClient(Base):
  __tablename__ = 'subworldclient'

  id = Column(Integer, primary_key=True)
  name = Column(String(20)) 
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL

  # for Python
  client = relationship("Client", backref=backref("client_subworldClient"))
 
  def __init__(self, name, client_id):
    self.name = name
    self.client_id = client_id

  def __repr__(self):
    return "<SubworldClient('%s', '%d')>" % (self.name, self.client_id) 

class SubworldFile(Base):
  __tablename__ = 'subworldfile'
 
  id = Column(Integer, primary_key=True)
  name = Column(String(20)) 
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  session_id = Column(Integer)
  speech_type = Column(Enum('p','l','r','f'))
  shot_id = Column(Integer)

  # for Python
  client = relationship("Client", backref=backref("client_subworldFile"))
 
  def __init__(self, name, client_id, session_id, speech_type, shot_id):
    self.name = name
    self.client_id = client_id
    self.session_id = session_id
    self.speech_type = speech_type
    self.shot_id = shot_id

  def __repr__(self):
    return "<SubworldFile('%s', '%d', '%d', '%d', '%d')>" % (self.name, self.client_id, self.session_id, self.speech_type, self.shot_id) 
