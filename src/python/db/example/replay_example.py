#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Jun 28 12:09:21 2011 +0200
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

"""Examplifies how to use the replay attack database.
"""

import bob

db = bob.db.replay.Database()
rawdata = "/idiap/group/replay/database/protocols"
processed = "gray27"
extension = ".jpg"

# Accesses the real-accesses and attacks for the PRINT-ATTACK protocol
data = db.files(directory=rawdata, extension='.mov', device='print',
    cls='real')
data.update(db.files(directory=rawdata, extension='.mov', device='print', 
    cls='attack'))

# Example: extracts frame 27 from each of the data, save it as a gray-scale

for key, value in data:
  print "Processing %s" % value
  snapshot = bob.database.VideoReader(value)[27]
  gray = snapshot.grayAs()
  bob.ip.rgb_to_gray(snapshot, gray)
  db.save_one(key, gray, processed, extension)
