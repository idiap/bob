#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  2 Aug 20:09:06 2011 

"""Transcode a model file to a binary or compressed format upon user request.
"""

import argparse
import bob

__epilog__ = """Example usage:

1. Transcode from a native, text format to compressed binary format:

  visioner_transcode.py Facial.MCT9 Facial.MCT9.vbgz

2. Transcode from a binary compressed format to compressed text

  visioner_transcode.py Facial.MCT9.vbgz Facial.MCT9.gz
"""

def main():
  
  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("input", metavar='FILE', type=str,
      help="the input filename")
  parser.add_argument("output", metavar='FILE', type=str,
      help="the output filename")
  
  args = parser.parse_args()

  bob.visioner.model_transcode(args.input, args.output)

if __name__ == '__main__':
  main()
