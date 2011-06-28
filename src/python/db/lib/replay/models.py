#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 11 May 18:52:38 2011

"""Table models and functionality for the Replay Attack DB.
"""

from sqlalchemy import Column, Integer, Enum, String, ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'
  
  id = Column(Integer, primary_key=True)
  set = Column(Enum('train', 'devel', 'test'))

  def __init__(self, id, set):
    self.id = id
    self.set = set

  def __repr__(self):
    return "<Client('%s', '%s')>" % (self.id, self.set)

class File(Base):
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  light = Column(Enum('controlled', 'adverse'))

  # for Python
  client = relationship(Client, backref=backref('realaccesses', order_by=id))

  def __init__(self, client, path, light):
    self.client = client
    self.path = path
    self.light = light

  def __repr__(self):
    print "<File('%s')>" % self.path

class RealAccess(Base):
  __tablename__ = 'realaccess'

  id = Column(Integer, primary_key=True)
  file_id = Column(Integer, ForeignKey('file.id')) # for SQL
  purpose = Column(Enum('authenticate', 'enroll'))
  take = Column(Integer)

  # for Python
  file = relationship(File, backref=backref('realaccess', order_by=id))

  def __init__(self, file, purpose, take):
    self.file = file
    self.purpose = purpose
    self.take = take

  def __repr__(self):
    return "<RealAccess('%s')>" % (self.file.path)

class Attack(Base):
  __tablename__ = 'attack'

  id = Column(Integer, primary_key=True)
  file_id = Column(Integer, ForeignKey('file.id')) # for SQL
  attack_support = Column(Enum('fixed', 'hand'))
  attack_device = Column(Enum('print', 'mobile', 'highdef', 'mask'))
  sample_type = Column(Enum('video', 'photo'))
  sample_device = Column(Enum('mobile', 'highdef'))

  # for Python
  file = relationship(File, backref=backref('attack', order_by=id))

  def __init__(self, file, attack_support, attack_device, sample_type, sample_device):
    self.file = file
    self.attack_support = attack_support
    self.attack_device = attack_device
    self.sample_type = sample_type
    self.sample_device = sample_device

  def __repr__(self):
    return "<Attack('%s')>" % (self.file.path)
