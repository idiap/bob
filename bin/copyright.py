#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 22 Nov 08:05:58 2011 

"""Copyrights and makes sure the given source file has a proper file header
"""

import os
import sys
import re
import subprocess
import time
import shutil
import unicodedata

COMMENT_START = re.compile(r'\s*/\*\*?\s*$')
COMMENT_END = re.compile(r'.*\*\/$')
FILEFIX = re.compile(r'^(\.\/)?src\/')
DOXY_FIELD = re.compile(r'^\s*\*\s*@(?P<field>[^\*\s]+)\s*.')

HEADER_START = '''\
/**
 * @file %(file)s
 * @date %(date)s
 * @author %(author)s
 *
'''

COPYRIGHT = '''\
 * Copyright (C) %(year)d Idiap Reasearch Institute, Martigny, Switzerland
'''

LICENSE = '''\
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */'''

def remove_diacritic(input):
  """Accepts a unicode string, and returns a normal string (bytes in Python 3)
  without any diacritical marks.
  """
  return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def get_gitinfo (filename):
  """Returns the git information concerning a file"""

  cmd = ['git', 'log', '--reverse', filename]
  out = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]

  retval = {}

  for line in out.split('\n'):
    if line.find('Author') == 0: 
      retval['author'] = line.split(' ', 1)[1].strip().decode('utf8')
    if line.find('Date') == 0: 
      retval['date'] = line.split(' ', 1)[1].strip().decode('utf8')
      break

  return retval

def load_parts (filename, max=3):
  """Returns the full start comment section of a file"""

  f = open(filename, 'rt')
  comment = []

  line = f.readline()

  tryfor = max
  while not COMMENT_START.match(line) and tryfor:
    tryfor -= 1
    if tryfor <= 0: 
      #print "Header marker not found at '%s' (tried %d first lines)" % \
      #    (filename, max)
      break

  comment_found = False
  if COMMENT_START.match(line): #found start of standard header
    comment_found = True
    while line:
      if not COMMENT_END.match(line):
        comment.append(line)
        line = f.readline()
      else:
        comment.append(line)
        break

  if not comment_found: #must rewind the file
    f.seek(0)

  contents = [k.decode('utf-8') for k in f.readlines()]

  return comment, contents

def create_new_comment(filename):

  #gather all necessary information
  info = get_gitinfo(filename)
  info['file'] = FILEFIX.sub('', filename)
  info['year'] = time.localtime().tm_year

  #create header
  retval = HEADER_START % info 
  retval += COPYRIGHT % info
  retval += LICENSE

  return retval.split('\n')

def rewrite_comment(filename, comment):

  #extracts parts to save...
  saved = []
  for line in comment:
    m = DOXY_FIELD.match(line)
    if m and m.group('field') in ('file', 'author', 'date'): continue
    if COMMENT_START.match(line) or COMMENT_END.match(line): continue
    s = line.strip()
    if s == '*': continue
    if s.find('* Copyright (C)') == 0: break
    saved.append(line.rstrip().decode('utf-8'))

  #gather all necessary information
  info = get_gitinfo(filename)
  info['file'] = FILEFIX.sub('', filename)
  info['year'] = time.localtime().tm_year

  #create header
  retval = HEADER_START % info 
  for k in saved: retval += k + '\n'
  if saved: retval += ' *\n'
  retval += COPYRIGHT % info
  retval += LICENSE

  return retval.split('\n')

def rewrite (filename, comment, contents, output):

  if not comment:
    new_comment = create_new_comment(filename)
  else:
    new_comment = rewrite_comment(filename, comment)

  f = open(output, 'wt')
  #f.writelines([remove_diacritic(k).encode('utf8') + '\n' for k in new_comment])
  #f.writelines([remove_diacritic(k).encode('utf8') for k in contents])
  f.writelines([k.encode('utf8') + '\n' for k in new_comment])
  f.writelines([k.encode('utf8') for k in contents])
  f.close()

def main ():

  if len(sys.argv) != 2:
    print "usage: %s <filename>" % os.path.basename(sys.argv[0])
    sys.exit(1)

  comment, contents = load_parts(sys.argv[1])
  shutil.copyfile(sys.argv[1], sys.argv[1] + '~')
  print "%s -> %s~" % (sys.argv[1], sys.argv[1])
  rewrite(sys.argv[1], comment, contents, sys.argv[1])
  print "%s: OK" % (sys.argv[1])

if __name__ == '__main__':
  main()
