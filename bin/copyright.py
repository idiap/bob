#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Nov 22 13:10:27 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Copyrights and makes sure the given source file has a proper file header. To
use this program, just do like this:

  $ for f in $(find src -type f); do bin/copyright.py $f; done
"""

import os
import sys
import re
import subprocess
import time
import shutil
import unicodedata

CXX_COMMENT_START = re.compile(r'\s*/\*\*?\s*$')
CXX_COMMENT_END = re.compile(r'.*\*\/$')
CXX_FILEFIX = re.compile(r'^(\.\/)?src\/')
CXX_DOXY_FIELD = re.compile(r'^\s*\*\s*@(?P<field>[^\*\s]+)\s*.')

PY_COMMENT = re.compile(r'^\s*#.*$')

RST_COMMENT = re.compile(r'^\.\..*$')

CXX_HEADER_START = '''\
/**
 * @file %(file)s
 * @date %(date)s
 * @author %(author)s
 *
'''

PY_HEADER_START = '''\
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# %(author)s
# %(date)s
#
'''

RST_HEADER_START = '''\
.. vim: set fileencoding=utf-8 :
.. %(author)s
.. %(date)s
.. 
'''

COPYRIGHT = '''\
%(cs)s Copyright (C) 2011-%(year)s Idiap Reasearch Institute, Martigny, Switzerland
'''

CXX_COPYRIGHT = COPYRIGHT % {'cs': ' *', 'year': '%(year)d'}
PY_COPYRIGHT = COPYRIGHT % {'cs': '#', 'year': '%(year)d'}
RST_COPYRIGHT = COPYRIGHT % {'cs': '..', 'year': '%(year)d'}

LICENSE = '''\
%(cs)s 
%(cs)s This program is free software: you can redistribute it and/or modify
%(cs)s it under the terms of the GNU General Public License as published by
%(cs)s the Free Software Foundation, version 3 of the License.
%(cs)s 
%(cs)s This program is distributed in the hope that it will be useful,
%(cs)s but WITHOUT ANY WARRANTY; without even the implied warranty of
%(cs)s MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%(cs)s GNU General Public License for more details.
%(cs)s 
%(cs)s You should have received a copy of the GNU General Public License
%(cs)s along with this program.  If not, see <http://www.gnu.org/licenses/>.
%(cse)s'''

CXX_LICENSE = LICENSE % {'cs': ' *', 'cse': ' */'}
PY_LICENSE = LICENSE % {'cs': '#', 'cse': ''}
RST_LICENSE = LICENSE % {'cs': '..', 'cse': ''}

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

def cxx_load_parts (filename, max=3):
  """Returns the full start comment section of a file"""

  f = open(filename, 'rt')
  comment = []

  line = f.readline()

  tryfor = max
  while not CXX_COMMENT_START.match(line) and tryfor:
    tryfor -= 1
    if tryfor <= 0: 
      #print "Header marker not found at '%s' (tried %d first lines)" % \
      #    (filename, max)
      break

  comment_found = False
  if CXX_COMMENT_START.match(line): #found start of standard header
    comment_found = True
    while line:
      if not CXX_COMMENT_END.match(line):
        comment.append(line)
        line = f.readline()
      else:
        comment.append(line)
        break

  if not comment_found: #must rewind the file
    f.seek(0)

  contents = [k.decode('utf-8') for k in f.readlines()]

  return comment, contents

def py_load_parts (filename, max=1):
  """Returns the full start comment section of a file"""

  f = open(filename, 'rt')
  comment = []

  line = f.readline()

  tryfor = max
  while not PY_COMMENT.match(line) and tryfor:
    tryfor -= 1
    if tryfor <= 0: 
      print "Header marker not found at '%s' (tried %d first lines)" % \
          (filename, max)
      break

  comment_found = False
  if PY_COMMENT.match(line): #found start of standard header
    comment_found = True
    while line:
      if PY_COMMENT.match(line):
        comment.append(line)
        line = f.readline()
      else:
        break

  if not comment_found: #must rewind the file
    f.seek(0)

  contents = [k.decode('utf-8') for k in f.readlines()]

  return comment, contents

def rst_load_parts (filename, max=1):
  """Returns the full start comment section of a file"""

  f = open(filename, 'rt')
  comment = []

  line = f.readline()

  tryfor = max
  while not RST_COMMENT.match(line) and tryfor:
    tryfor -= 1
    if tryfor <= 0: 
      print "Header marker not found at '%s' (tried %d first lines)" % \
          (filename, max)
      break

  comment_found = False
  if RST_COMMENT.match(line): #found start of standard header
    comment_found = True
    while line:
      if RST_COMMENT.match(line):
        comment.append(line)
        line = f.readline()
      else:
        break

  if not comment_found: #must rewind the file
    f.seek(0)

  contents = [k.decode('utf-8') for k in f.readlines()]

  return comment, contents

def cxx_create_new_comment(filename):

  #gather all necessary information
  info = get_gitinfo(filename)
  info['file'] = CXX_FILEFIX.sub('', filename)
  info['year'] = time.localtime().tm_year

  #create header
  retval = CXX_HEADER_START % info 
  retval += CXX_COPYRIGHT % info
  retval += CXX_LICENSE

  return retval.split('\n')

def py_create_new_comment(filename):

  #gather all necessary information
  info = get_gitinfo(filename)
  info['year'] = time.localtime().tm_year

  #create header
  retval = PY_HEADER_START % info 
  retval += PY_COPYRIGHT % info
  retval += PY_LICENSE

  return retval.split('\n')

def rst_create_new_comment(filename):

  #gather all necessary information
  info = get_gitinfo(filename)
  info['year'] = time.localtime().tm_year

  #create header
  retval = RST_HEADER_START % info 
  retval += RST_COPYRIGHT % info
  retval += RST_LICENSE

  return retval.split('\n')

def cxx_rewrite_comment(filename, comment):

  #extracts parts to save...
  saved = []
  for line in comment:
    m = CXX_DOXY_FIELD.match(line)
    if m and m.group('field') in ('file', 'author', 'date'): continue
    if CXX_COMMENT_START.match(line) or CXX_COMMENT_END.match(line): continue
    s = line.strip()
    if s == '*': continue
    if s.find('* Copyright (C)') == 0: break
    saved.append(line.rstrip().decode('utf-8'))

  #gather all necessary information
  info = get_gitinfo(filename)
  info['file'] = CXX_FILEFIX.sub('', filename)
  info['year'] = time.localtime().tm_year

  #create header
  retval = CXX_HEADER_START % info 
  for k in saved: retval += k + '\n'
  if saved: retval += ' *\n'
  retval += CXX_COPYRIGHT % info
  retval += CXX_LICENSE

  return retval.split('\n')

def cxx_rewrite (filename, comment, contents, output):

  if not comment: new_comment = cxx_create_new_comment(filename)
  else: new_comment = cxx_rewrite_comment(filename, comment)

  f = open(output, 'wt')
  #f.writelines([remove_diacritic(k).encode('utf8') + '\n' for k in new_comment])
  #f.writelines([remove_diacritic(k).encode('utf8') for k in contents])
  f.writelines([k.encode('utf8') + '\n' for k in new_comment])
  f.writelines([k.encode('utf8') for k in contents])
  f.close()

def py_rewrite (filename, comment, contents, output):

  new_comment = py_create_new_comment(filename)

  f = open(output, 'wt')
  #f.writelines([remove_diacritic(k).encode('utf8') + '\n' for k in new_comment])
  #f.writelines([remove_diacritic(k).encode('utf8') for k in contents])
  f.writelines([k.encode('utf8') + '\n' for k in new_comment])
  f.writelines([k.encode('utf8') for k in contents])
  f.close()

def rst_rewrite (filename, comment, contents, output):

  new_comment = rst_create_new_comment(filename)

  f = open(output, 'wt')
  #f.writelines([remove_diacritic(k).encode('utf8') + '\n' for k in new_comment])
  #f.writelines([remove_diacritic(k).encode('utf8') for k in contents])
  f.writelines([k.encode('utf8') + '\n' for k in new_comment])
  f.writelines([k.encode('utf8') for k in contents])
  f.close()

def find_filetype(filename):
  """Returns the most probable filetype given the file name"""

  ext = os.path.splitext(sys.argv[1])[1]
  if filename == 'CMakeLists.txt': return 'cmake'
  elif ext in ('.sh', '.csh'): return 'shell'
  elif ext in ('.c', '.cc', '.cxx', '.C', '.CC', '.cpp', '.h', '.hh', '.H',
      '.hpp', '.HH'): return 'c'
  elif ext in ('.py', '.PY'): return 'python'
  elif ext in ('.rst',): return 'rst'
  
  print "warning: unsupported extension: %s" % filename
  return None

def main ():

  if len(sys.argv) != 2:
    print "usage: %s <filename>" % os.path.basename(sys.argv[0])
    sys.exit(1)

  filetype = find_filetype(sys.argv[1])

  if filetype == 'c': comment, contents = cxx_load_parts(sys.argv[1])
  elif filetype in ('python',): comment, contents = py_load_parts(sys.argv[1])
  elif filetype in ('rst',): comment, contents = rst_load_parts(sys.argv[1])

  shutil.copyfile(sys.argv[1], sys.argv[1] + '~')
  print "%s -> %s~" % (sys.argv[1], sys.argv[1])

  if filetype == 'c': cxx_rewrite(sys.argv[1], comment, contents, sys.argv[1])
  elif filetype in ('python',): py_rewrite(sys.argv[1], comment, contents, sys.argv[1])
  elif filetype in ('rst',): rst_rewrite(sys.argv[1], comment, contents, sys.argv[1])

  print "%s: OK" % (sys.argv[1])

if __name__ == '__main__':
  main()
