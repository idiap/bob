#!/idiap/home/aanjos/work/torch/main/bin/shell.py -- python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 18 May 09:59:13 2011 

"""Examplifies how to use the replay attack database.
"""

import torch

db = torch.db.replay.Database()
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
  snapshot = torch.database.VideoReader(value)[27]
  gray = snapshot.grayAs()
  torch.ip.rgb_to_gray(snapshot, gray)
  db.save_one(key, gray, processed, extension)
