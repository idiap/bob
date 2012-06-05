#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Thu May 24 10:41:42 CEST 2012
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

"""The Labeled Faces in the Wild (LFW) face database. Please refer to 
http://vis-www.cs.umass.edu/lfw for information how to get a copy of it.

The LFW database provides two different sets (called "views". The first one, called "view1"
is to be used for optimizing meta-parameters of your algorithm. 
When querying the database, please use ``protocol='view1'`` to get the data only for this view.
Please note that there is only a ``groups='dev'`` group, but no ``'eval'`` in ``view1``.

The second view is split up into 10 different "folds". According to http://vis-www.cs.umass.edu/lfw
in each fold 9/10 of the database are used for training, and one for evaluation.
In **this implementation** of the LFW database, 8/10 of the data is used for training (``groups='world'``),
1/10 are used for development (to estimate a threshold; ``groups='dev'``) and the last 1/10 is finally
used to evaluate the system (``groups='eval'``). 

To compute recognition results, please execute experiments on all 10 protocols (``protocol='fold1'`` ... ``protocol='fold10'``) 
and average the resulting classification results (cf. http://vis-www.cs.umass.edu/lfw for details on scoring).

The design of this implementation differs slightly with the one from http://vis-www.cs.umass.edu/lfw.
Originally, only lists of image pairs are provided by the creators of the LFW database. 
To be consistent with other Bob databases, here the lists are split up into files to be enrolled, and probe files.
The files to be enrolled are always the first file in the pair, while the second pair item is used as probe. 

.. note::
  
  When querying probe files, please **always** query probe files for a specific model id:
  ``files(..., purposes = 'probe', model_ids = (model_id,))``
  
When querying training files ``files(..., groups='world')``, you will automatically end up with the "unrestricted configuration". 
When you want to respect the "image restricted configuration" (cf. README on http://vis-www.cs.umass.edu/lfw),
please either:

* query image pairs ``pairs(..., groups = 'world')``
* query the files that belong to the pairs, via ``files(..., groups='world', subworld='restricted')``

Of course, you can also query the "dev" and "eval" pairs if you prefer to stick to image pairs only.

.. note::

  The pairs that are provided using the ``pairs`` function, and the files provided by the ``files`` function 
  (see note above) correspond to the identical model/probe pairs. Hence, either of the two approaches should give the same recognition results.
"""

def dbname():
  """Calculates my own name automatically."""
  import os
  return os.path.basename(os.path.dirname(__file__))

from .query import Database
from .commands import add_commands
