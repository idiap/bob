#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 28 Jun 2011 12:54:06 CEST 

"""Contains a set of management utilities for a centralized driver script.
"""
  
import os
import sys
import time

def files_all(args):
  """Executes all the files commands from individual databases"""
  
  for name in [k.name() for k in args.modules]:
    parsed = args.parser.parse_args([name, 'files'])
    parsed.func(parsed)

def create_all(args):
  """Executes all the default create commands from individual databases"""
 
  errors = 0
  databases = 0
  total_start = time.time()

  sqlite_dbs = [k.name() for k in args.modules if k.type() in ('sqlite',)]

  if args.verbose >= 1:
    print '### Running %d SQLite database creation commands...' % len(sqlite_dbs)

  for name in sqlite_dbs:
    
    start_time = time.time()
    databases += 1

    parsed = args.parser.parse_args([name, 'create'])
    parsed.recreate = args.recreate
    parsed.verbose = args.verbose

    if args.verbose >= 1:
      print '>>> Creating "%s" SQLite database...' % name

    try:
      parsed.func(parsed)

    except:
      
      errors += 1

      if args.keep_going:
        if args.verbose >= 1:
          print 'Warning: Error while creating "%s" SQLite database' % name
        __import__('traceback').print_exc()
        if args.verbose >= 1:
          print '*** Keep going on user request...'

      else:
        raise

    finally:

      if args.verbose >= 1:
        print '<<< Finished creation of "%s" SQLite database (%.2f seconds).' % \
            (name, time.time()-start_time)

  if args.verbose >= 1:
    print '### %d SQLite databases created in %.2f seconds, %d errors' % \
        (databases, time.time()-total_start, errors)

def version_all(args):
  """Executes all the default version commands from individual databases"""
  
  for name in [k.name() for k in args.modules]:
    parsed = args.parser.parse_args([name, 'version'])
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

  from .driver import files_command, version_command

  # creates a top-level parser for this database
  top_level = top_subparser.add_parser('all',
      help="Drive commands to all (above) databases in one shot",
      description="Using this special database you can command the execution of available commands to all other databases at the same time.")

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  files_parser = files_command(subparsers)
  files_parser.set_defaults(func=files_all)
  files_parser.set_defaults(parser=parser)
  files_parser.set_defaults(modules=modules)

  create_parser = subparsers.add_parser('create',
      help="create all databases with default settings")
  create_parser.add_argument('-k', '--keep-going',
      action='store_true', default=False,
      help="If set, will survive a database creation failure and keep-on trying to create other databases")
  create_parser.add_argument('-R', '--recreate',
      action='store_true', default=False,
      help="If set, I'll first erase the current database")
  create_parser.add_argument('-v', '--verbose', action='count',
      help="Do SQL operations in a verbose way")
  create_parser.set_defaults(func=create_all)
  create_parser.set_defaults(parser=parser)
  create_parser.set_defaults(modules=modules)

  version_parser = version_command(subparsers)
  version_parser.set_defaults(func=version_all)
  version_parser.set_defaults(parser=parser)
  version_parser.set_defaults(modules=modules)

def create_parser(**kwargs):
  """Creates a parser for the central manager taking into consideration the
  options for every module that can provide those."""

  import pkg_resources
  import argparse
  import imp

  parser = argparse.ArgumentParser(**kwargs)
  subparsers = parser.add_subparsers(title='databases')

  # for external entries
  for entrypoint in pkg_resources.iter_entry_points('bob.db'):
    plugin = entrypoint.load()

  # at this point we should have loaded all databases
  from .driver import Interface
  all_modules = []
  for plugin in Interface.__subclasses__():
    driver = plugin()
    driver.add_commands(subparsers)
    all_modules.append(driver)

  add_all_commands(parser, subparsers, all_modules) #inserts the master driver

  return parser
