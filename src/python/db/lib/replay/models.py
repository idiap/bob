#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 11 May 18:52:38 2011

"""Table models and functionality for the Replay Attack DB.
"""

from sqlalchemy import Table, Column, Integer, String, ForeignKey
from ..sqlalchemy_migration import Enum, relationship
from sqlalchemy.orm import backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Client(Base):
  __tablename__ = 'client'

  set_choices = ('train', 'devel', 'test')
  
  id = Column(Integer, primary_key=True)
  set = Column(Enum(*set_choices))

  def __init__(self, id, set):
    self.id = id
    self.set = set

  def __repr__(self):
    return "<Client('%s', '%s')>" % (self.id, self.set)

class File(Base):
  __tablename__ = 'file'

  light_choices = ('controlled', 'adverse')

  id = Column(Integer, primary_key=True)
  client_id = Column(Integer, ForeignKey('client.id')) # for SQL
  path = Column(String(100), unique=True)
  light = Column(Enum(*light_choices))

  # for Python
  client = relationship(Client, backref=backref('files', order_by=id))

  def __init__(self, client, path, light):
    self.client = client
    self.path = path
    self.light = light

  def __repr__(self):
    print "<File('%s')>" % self.path

# Intermediate mapping from RealAccess's to Protocol's
realaccesses_protocols = Table('realaccesses_protocols', Base.metadata,
    Column('realaccess_id', Integer, ForeignKey('realaccess.id')),
    Column('protocol_id', Integer, ForeignKey('protocol.id')),
    )

# Intermediate mapping from Attack's to Protocol's
attacks_protocols = Table('attacks_protocols', Base.metadata,
    Column('attack_id', Integer, ForeignKey('attack.id')),
    Column('protocol_id', Integer, ForeignKey('protocol.id')),
    )

class Protocol(Base):
  __tablename__ = 'protocol'

  id = Column(Integer, primary_key=True)
  name = Column(String(20), unique=True)

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return "<Protocol('%s')>" % (self.name,)

class RealAccess(Base):
  __tablename__ = 'realaccess'

  purpose_choices = ('authenticate', 'enroll')

  id = Column(Integer, primary_key=True)
  file_id = Column(Integer, ForeignKey('file.id')) # for SQL
  purpose = Column(Enum(*purpose_choices))
  take = Column(Integer)

  # for Python
  file = relationship(File, backref=backref('realaccess', order_by=id))
  protocols = relationship("Protocol", secondary=realaccesses_protocols,
      backref='realaccesses')

  def __init__(self, file, purpose, take):
    self.file = file
    self.purpose = purpose
    self.take = take

  def __repr__(self):
    return "<RealAccess('%s')>" % (self.file.path)

class Attack(Base):
  __tablename__ = 'attack'

  attack_support_choices = ('fixed', 'hand')
  attack_device_choices = ('print', 'mobile', 'highdef', 'mask')
  sample_type_choices = ('video', 'photo')
  sample_device_choices = ('mobile', 'highdef')

  id = Column(Integer, primary_key=True)
  file_id = Column(Integer, ForeignKey('file.id')) # for SQL
  attack_support = Column(Enum(*attack_support_choices))
  attack_device = Column(Enum(*attack_device_choices))
  sample_type = Column(Enum(*sample_type_choices))
  sample_device = Column(Enum(*sample_device_choices))

  # for Python
  file = relationship(File, backref=backref('attack', order_by=id))
  protocols = relationship("Protocol", secondary=attacks_protocols,
      backref='attacks')

  def __init__(self, file, attack_support, attack_device, sample_type, sample_device):
    self.file = file
    self.attack_support = attack_support
    self.attack_device = attack_device
    self.sample_type = sample_type
    self.sample_device = sample_device

  def __repr__(self):
    return "<Attack('%s')>" % (self.file.path)
