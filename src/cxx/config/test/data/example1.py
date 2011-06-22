#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sun  6 Mar 17:09:52 2011 

"""Example torch configuration
"""

import os, torch

param1 = "my test string"

param2 = 3.1416

param3 = torch.core.array.int16_3(range(24), (2,3,4))

curdir = os.path.dirname(__file__)

param4 = torch.io.Array(os.path.join(curdir, "arrays.bin"))

param5 = torch.io.Arrayset(os.path.join(curdir, "arrays.bin"))
