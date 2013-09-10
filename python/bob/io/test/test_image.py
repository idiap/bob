#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Wed Aug 14 12:27:57 CEST 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

"""Runs some image tests
"""

import os
import sys
import numpy

from ...test import utils as testutils

# These are some global parameters for the test.
PNG_INDEXED_COLOR = testutils.datafile('img_indexed_color.png', __name__)

def test_png_indexed_color():

  # Read an indexed color PNG image, and compared with hardcoded values
  from .. import load
  img = load(PNG_INDEXED_COLOR)
  assert img.shape == (3,22,32)
  assert img[0,0,0] == 255
  assert img[0,17,17] == 117
