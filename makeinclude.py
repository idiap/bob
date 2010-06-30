#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 29 Jun 2010 14:58:41 CEST 

"""Creates an unified include file for all torch5spro headers. """

import os,sys,fnmatch,time

def main():
  if len(sys.argv) < 2:
    print 'usage: %s <output-header-file> <excludes>' % sys.argv[0]
    sys.exit(1)

  dirname = os.path.dirname(sys.argv[1])
  output = os.path.basename(sys.argv[1])

  headers = get_headers(dirname, [output] + sys.argv[2:])
  write_header(sys.argv[1], headers)

def get_headers(dir, excludes):
  """Gets all files ending in '.h' from the directory, recursively, except for
  what is defined in the input argument "excludes"."""
  retval = []
  for (path, dirs, files) in os.walk(dir):
    for f in fnmatch.filter(files, '*.h'):
      if f in excludes: continue
      sub = path.replace(dir+os.sep, '')
      retval.append(os.path.join(sub, f))
  return retval

def write_header(output, headers):
  """Writes a new header file that incorporates all existing ones."""
  f = open(output, 'wt')
  f.write('/* This file was automatically generated -- DO NOT CHANGE IT */\n')
  f.write('/* Date: %s */\n\n' % time.asctime())
  f.write('#ifndef __TORCH5SPROC_H__\n')
  f.write('#define __TORCH5SPROC_H__\n\n')
  f.writelines(['#include "%s"\n' % k for k in headers])
  f.write('\n#endif /* __TORCH5SPROC_H__ */\n')
  f.close()

if __name__ == '__main__':
  main()

