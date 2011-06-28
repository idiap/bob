#!/idiap/group/torch5spro/nightlies/last/bin/shell.py -- python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 14:02:28 2011 

"""This program demonstrates how to dump flexible file lists for the replay
attack database.
"""

import os
import sys

def get_options():
  """Processes user options and arguments, makes sure all is ok."""
  
  import optparse

  usage = 'usage: %s [options]' % os.path.basename(sys.argv[0])

  parser = optparse.OptionParser(usage=usage, description=__doc__)

  parser.add_option('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%default')")
  parser.add_option('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%default')")
  parser.add_option('-s', '--support', dest="support", default='', help="if given, this value will limit the output files to those using this type of attack support. Valid values are 'fixed' or 'hand'. (defaults to '%default')", choices=('fixed', 'hand', ''))
  parser.add_option('-c', '--device', dest="device", default='', help="if given, this value will limit the output files to those using this type of device for attacks. Valid values are 'print', 'mobile' or 'highdef'. (defaults to '%default')", choices=('print', 'mobile', 'highdef', ''))

  options, args = parser.parse_args()

  if len(args) != 0:
    parser.error("this program does not take positional arguments")

  return options

if __name__ == '__main__':
  from replay import Database
  options = get_options()
  db = Database()

  sets = ('train', 'devel', 'test')

  print "Real-accesses:"
  for set in sets:
    r = db.files(directory=options.directory,
        extension=options.extension, device=options.device,
        support=options.support, cls='real', groups=set)
    print '  ' + set
    for id, f in r: print '    %d %s' % (id, f)
  
  print "Attacks:"
  for set in sets:
    r = db.files(directory=options.directory,
        extension=options.extension, device=options.device,
        support=options.support, cls='attack', groups=set)
    print '  ' + set
    for id, f in r: print '    %d %s' % (id, f)
