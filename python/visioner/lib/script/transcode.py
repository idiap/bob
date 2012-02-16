#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Aug 2 20:42:51 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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
