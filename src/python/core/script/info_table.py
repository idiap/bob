#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 27 Apr 22:59:28 2011 

"""Prints the version of Torch and exits.
"""

import os
import sys
import torch

print "Torch5spro: %s (%s)" % (os.environ['TORCH_VERSION'], os.environ['TORCH_PLATFORM'])
print torch.core.version_string()
sys.exit(0)
