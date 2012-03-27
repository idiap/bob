.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 27 Mar 2012
.. 
.. Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

========================================
 Contributing to the |project| project
========================================

.. todo::
  Update the instructions below (e.g. are "tagging and branching" instructions relevant here? Also, developers will probably need to submit patches rather than git push).

Tagging and Branching
=====================

For |project| development, we adopt the following scheme for tagging and
branching:

* Tagging can be done on the master branch when milestones are reached,
  important steps have been accomplished or in the event of releases;
* Branching is only allowed when releasing an new version of Torch. Avoid
  pushing local branches forked for your private development into the main
  repository. Only push branches that should belong to the project history.


Pushing your changes
====================

If you are an authorized developer, you can push your changes back into the
central repository. Please, **make sure you comment your commits** and have
your global email and name configurations set as described above. Also
double-check that 

1. Your changes compile without problems

2. There must always be unit tests that cover new code. New code is not acceptable otherwise; 

3. New code must always come with extensive documentation. We have put in place tools to assure that code that breaks these rules can be detected where possible.
   
Then, from your cloned repository, do:

.. code-block:: sh

  $ git push


.. include:: links.rst

