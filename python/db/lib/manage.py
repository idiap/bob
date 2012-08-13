#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 28 Jun 2011 12:54:06 CEST 

"""Contains a set of management utilities for a centralized driver script.
"""
  
import os
import sys

def location_all(args):
  """Executes all the location commands from individual databases"""
  
  for name, module in args.modules:
    parsed = args.parser.parse_args([name, 'location'])
    parsed.with_protocol = args.with_protocol
    parsed.func(parsed)

def get_all(args):
  """Executes all the get commands from individual databases"""
  
  for name, module in args.modules:
    parsed = args.parser.parse_args([name, 'get', args.directory[0]])
    parsed.verbose = args.verbose
    parsed.func(parsed)

def put_all(args):
  """Executes all the put commands from individual databases"""
  
  for name, module in args.modules:
    parsed = args.parser.parse_args([name, 'put', args.directory[0]])
    parsed.verbose = args.verbose
    parsed.func(parsed)

def create_all(args):
  """Executes all the default create commands from individual databases"""
  
  for name, module in args.modules:
    parsed = args.parser.parse_args([name, 'create'])
    parsed.recreate = args.recreate
    parsed.verbose = args.verbose
    parsed.func(parsed)

def add_all_commands(parser, top_subparser, modules):
  """Adds a subset of commands all databases must comply to and that can be
  triggered for all databases. This special "database" is just a mask to
  executed all other databases commands in a single run. For details on
  argparse, please visit:

  http://docs.python.org/dev/library/argparse.html

  The strategy assumed here is that each command will have its own set of
  options that are relevant to that command. So, we just scan such commands and
  attach the options from those.
  """

  from .utils import location_command, put_command, get_command

  # creates a top-level parser for this database
  top_level = top_subparser.add_parser('all',
      help="Drive commands to all (above) databases in one shot",
      description="Using this special database you can command the execution of available commands to all other databases at the same time.")

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  location_parser = location_command(subparsers)
  location_parser.set_defaults(func=location_all)
  location_parser.set_defaults(parser=parser)
  location_parser.set_defaults(modules=modules)

  put_parser = put_command(subparsers)
  put_parser.set_defaults(func=put_all)
  put_parser.set_defaults(parser=parser)
  put_parser.set_defaults(modules=modules)

  get_parser = get_command(subparsers)
  get_parser.set_defaults(func=get_all)
  get_parser.set_defaults(parser=parser)
  get_parser.set_defaults(modules=modules)

  create_parser = subparsers.add_parser('create',
      help="create all databases with default settings")
  create_parser.add_argument('-R', '--recreate',
      action='store_true', default=False,
      help="If set, I'll first erase the current database")
  create_parser.add_argument('-v', '--verbose', action='append_const',
      const=1, help="Do SQL operations in a verbose way")
  create_parser.set_defaults(func=create_all)
  create_parser.set_defaults(parser=parser)
  create_parser.set_defaults(modules=modules)

def create_parser(**kwargs):
  """Creates a parser for the central manager taking into consideration the
  options for every module that can provide those."""

  import pkg_resources
  import argparse

  parser = argparse.ArgumentParser(**kwargs)
  subparsers = parser.add_subparsers(title='databases')

  dirname = os.path.dirname(__file__)
  all_modules = []

  for entrypoint in pkg_resources.iter_entry_points('bob.db'):
    plugin = entrypoint.load()

    # adds command directives to the manager, for this specific database
    if hasattr(plugin, 'add_commands'): plugin.add_commands(subparsers)

  add_all_commands(parser, subparsers, all_modules) #inserts the master driver

  return parser
