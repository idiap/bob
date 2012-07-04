#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Wed Jul  4 14:12:51 CEST 2012
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

"""This is the Bob database entry for the AR face database. 

  The database can be downloaded from http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html 
  (though we were not able to contact the corresponding Professor).
  
  Our version of the database contains 3312 images from 136 persons, 76 men and 60 women.
  
  We split the database into several protocols that we have designed ourselves. 
  The identities are split up into three groups, 
  
  * the 'world' group for training your algorithm
  * the 'dev' group to optimize your algorithm parameters on
  * the 'eval' group that should only be used to report results   
  
  Additionally, there are different protocols:
  
  * 'expression': only the probe files with different facial expressions are selected
  * 'illumination': only the probe files with different illuminations are selected
  * 'occlusion': only the probe files with normal illumination and different accessories (scarf, sunglasses) are selected
  * 'occlusion_and_illumination': only the probe files with strong illumination and different accessories (scarf, sunglasses) are selected
  * 'all': all files are used as probe
  
  In any case, the images with neutral facial expression, neutral illumination and without accessories are used for enrollment.
"""

def dbname():
  """Calculates my own name automatically."""
  import os
  return os.path.basename(os.path.dirname(__file__))

from .query import Database
from .commands import add_commands
