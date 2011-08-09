#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  9 Aug 13:43:49 2011 

"""A program that replaces templates on template-enabled files"""

epilog = """Example usage:

  1) Replace "@tag1@" with "foo" and "@tag2@" with "bar" at file bla.tcc,
     dump output to an output file named bla_foo.cc

     $ %(prog)s -t @tag1@=foo -t @tag2@=bar bla.tcc bla_@tag1@.cc 
"""

import os
import sys
import optparse

def make_template_dict(templates):
  """Creates a template substitution dictionary"""
  retval = {}
  for k in templates:
    s = [t.strip() for t in k.split('=')]
    retval[s[0]] = s[1]
  return retval

def replace_templates(l, d):
  """Replaces templates on a text line"""
  for k, v in d.items(): l = l.replace(k, v)
  return l

def main():

  class MyParser(optparse.OptionParser):
    def format_epilog(self, formatter):
      return self.epilog

  parser = MyParser(description=__doc__, epilog=epilog % {'prog': os.path.basename(sys.argv[0])})

  parser.add_option('-t', '--template', dest='template', action='append',
      help='Templates that should be substituted on the file')

  opts, args = parser.parse_args()

  if len(args) != 2:
   parser.error("incorrect number of arguments -- see help message")

  tmpl_dict = make_template_dict(opts.template)

  args[1] = replace_templates(args[1], tmpl_dict)

  ifile = open(args[0], 'rt')
  ofile = open(args[1], 'wt')
  for line in ifile: ofile.write(replace_templates(line, tmpl_dict))
  ifile.close()
  ofile.close()

if __name__ == '__main__':
  main()
