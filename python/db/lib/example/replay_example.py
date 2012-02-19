#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Jun 28 12:09:21 2011 +0200
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

"""Examplifies how to use the replay attack database. Note that, for this
example, we create and destroy the output. You will probably not want to do
this in real-life (tm) ;-)
"""

import bob
import tempfile
import shutil

def main():

  db = bob.db.replay.Database()
  rawdata = "/idiap/group/replay/database/protocols"
    
  processed = tempfile.mkdtemp(prefix='bobtest_')
  extension = ".jpg"

  # Accesses the real-accesses and attacks for the PRINT-ATTACK protocol
  data = db.files(directory=rawdata, extension='.mov', protocol='print',
      cls='real')
  data.update(db.files(directory=rawdata, extension='.mov', protocol='print', 
      cls='attack'))

  # Example: extracts frame 27 from each of the data, save it as a gray-scale
  # stop after 5 examples...

  counter = 5
  for key, value in data.iteritems():
    print "Processing %s" % value
    snapshot = bob.io.VideoReader(value)[27]
    gray = bob.ip.rgb_to_gray(snapshot)
    db.save_one(key, gray, processed, extension)
    print "Saved gray-scale equivalent of %s at %s" % (value, processed)
    counter -= 1
    if counter <= 0: break

  shutil.rmtree(processed)
  print "Removed temporary directory %s" % processed

if __name__ == '__main__':
  main()
