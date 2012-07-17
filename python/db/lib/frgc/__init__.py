#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Fri Jul  6 16:45:41 CEST 2012
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

"""
The FRGC (Face Recognition Grand Challenge) is provided by the NIST webpage: http://face.nist.gov/frgc/
Please contact the responsible person to get your own copy of the database (be aware that it is HUGE)

In opposition to other databases, there is no .sql3-file for this database, but instead the XML lists provided in the database are used directly.
Please specify the directory to your copy of the FRGC database on creation of a ``bob.db.frgc.Database(directory)`` object.

In order to generate the eye position files (so that the database can be used like any other bob.db database),
please call ``bob_dbmanage.py frgc create-position-files --directory <YOUR_PATH>``.

In opposition to the original FRGC protocols, here only those image files and models that are required by the mask are used.
This saves some time and space, but ensures identical results.

.. warning ::

  Do not store the model ids between sessions. These model id's are generated 'on the fly' and might change between database sessions.
  
.. note ::
 
  During one database session, model ids are unique and stable. 

Enjoy!
"""

def dbname():
  """Calculates my own name automatically."""
  import os
  return os.path.basename(os.path.dirname(__file__))

from .query import Database
from .commands import add_commands
