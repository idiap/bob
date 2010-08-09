#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 09 Aug 2010 09:03:58 CEST 

"""Analyzes nightly builds in the current directory and produces an HTML table
that can be included in Trac via the IncludePlugin. The table contains
information about nightly builds, success information and problems found.
"""

import sys, os, time, re, re

epilog = """
Examples:

  If it is conveninent, just cd into the base nightly directory and type:
  $ %(prog)s

  To create the HTML code for a table in an arbitrary directory:
  $ %(prog)s --base-dir=<nightly-base-directory> 
""" % {'prog': os.path.basename(sys.argv[0])}

import os, sys

def parse_args():
  """Parses the command line input."""
  import optparse

  class MyParser(optparse.OptionParser):
    """Overwites the format_epilog() so we keep newlines..."""
    def format_epilog(self, formatter):
      return self.epilog

  dir = os.path.realpath(os.curdir)
  
  parser = MyParser(description=__doc__, epilog=epilog)
  parser.add_option("-b", "--base-dir", 
                    action="store",
                    dest="dir", 
                    default=dir,
                    help="Sets the base directory to a different value (defaults to %default)",
                   )

  options, arguments = parser.parse_args()

  options.dir = os.path.realpath(options.dir)

  return (options, arguments)

def find_builds(dir):
  """Finds all executed builds taking as root directory, 'dir'"""
  retval = {} 
  for (dir, dirnames, filenames) in os.walk(dir):
    if dir[-4:] != 'logs': continue #there is nothing elsewhere
    if 'status.py' not in filenames: continue #we need the status
    split = dir.split(os.path.sep)
    date = retval.get(split[-4], {})
    date[split[-2]] = os.path.join(dir, 'status.py')
    retval[split[-4]] = date
  return retval

def load_variables(fname):
  """Loads the status variables from the python file."""
  retval = {}
  execfile(fname, {}, retval)
  return retval

def count_problems(fname):
  """Counts warnings and errors in files."""
  error   = re.compile('error\s*:', re.I)
  warning = re.compile('warning\s*:', re.I)
  f = open(fname, 'rt')
  wcount = 0
  ecount = 0
  for line in f:
    if error.search(line): ecount += 1 
    if warning.search(line): wcount += 1 
  f.close()
  return (ecount, wcount)

def build_html_table(entries):
  """Builds an HTML table based on the found builds."""
  # Loads and resets information for the builds
  for k,v in entries.iteritems():
    for build in v.keys(): v[build] = load_variables(v[build])
  newentries = {}
  for k,v in entries.iteritems():
    newkey = min([val['timing']['start'] for key,val in v.iteritems()])
    newentries[newkey] = v
  entries = newentries

  # Now we build the table
  retval = ['<table style="margin-left: auto; margin-right: auto; font-family: sans-serif; text-align: center;">']
  retval.append('<tr>')
  retval.extend(['<th style="border: 1px black solid; padding: 5px; color: white; background-color: black;">%s</th>' % k for k in ('Date', 'Platform', 'Cmake', 'Build', 'Install', 'Documentation', 'Tests')])
  retval.append('</tr>')

  colors = {
            'success': '#8f8', 
            'warning': '#ff8', 
            'failed': '#f88', 
            'blocked': '#888'
           }
  ok_style = "font-weight: bold; font-size: 85%; color: green;"
  error_style = "font-weight: bold; font-size: 85%; color: red;"
  warning_style = "font-weight: bold; font-size: 85%; color: #550;"
  subscript_style = "color: #555; font-size: 70%; font-style: italic;"
  for k in sorted(entries.keys(), reverse=True):
    v = entries[k]
    start = time.localtime(k)
    retval.append('<tr style="border: 1px black solid;">')
    retval.append('<td rowspan="%d" style="border: 1px black solid; padding: 5px;">%s<br/><font style="%s">started: %s</font></td>' % (len(v), time.strftime("%d/%b/%Y", start), subscript_style, time.strftime("%H:%M:%S", start)))
    first = True
    for build in sorted(v.keys()):
      if not first: 
        retval.append('<tr>')
        first = False
      retval.append('<td style="border: 1px black solid; padding: 5px;">%s<br/><font style="%s">%s</font></td>' % (build, subscript_style, '%s-%s (%s)' % (v[build]['uname'][0], v[build]['uname'][2], v[build]['uname'][4])))
      for phase in ('cmake','make_all','make_install','doxygen','make_test'):
        status = v[build]['status'][phase][0]
        timing = v[build]['timing'][phase]
        log = '/'.join([v[build]['options']['log_prefix'], phase]) + '.txt'
        html_log = '../chrome/site/' + '/'.join(log.split('/')[-6:])
        extra_style = ok_style
        message = "No issues"
        errors, warnings = count_problems(log)
        if errors: 
          status = 'failed'
          extra_style = error_style
          message = "errors: %d; warnings: %d" % (errors, warnings)
        elif warnings: 
          status = 'warning'
          extra_style = warning_style
          message = 'warnings: %d' % (warnings)
        retval.append('<td style="background-color: %s; border: 1px black solid; padding: 5px;"><a href="%s">%s</a><br/><font style="%s">%s</font><br/><font style="%s">time: %.1f s</font></td>' % (colors[status], html_log, status.upper(), extra_style, message, subscript_style, timing))
      retval.append('</tr>')
    retval.append('</tr>')
  retval.append('<caption style="text-align: right; font-style: italic; font-size: 70%%; color: gray;">Last updated on %s</caption>' % time.asctime())
  retval.append('</table>')

  return '\n'.join(retval)

def main():

  options, arguments = parse_args()
  print build_html_table(find_builds(options.dir))

if __name__ == '__main__':
  main()
