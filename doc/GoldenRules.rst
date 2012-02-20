.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Tue Apr 26 18:35:34 2011 +0200
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
 Golden Rules for |project| Development
========================================

Developing against |project| should be a joy! Nevertheless, all software
projects need some ground rules (or **Golden Rules**) to guarantee minimal
usability and maintainability. Here are ours:

1. The latest version of our version control repository (sometimes referred as
   HEAD) on the main development branch, must **always** compile w/o
   problems. This means you **cannot** check-in broken code. You must **at
   least** make sure it compiles on all supported platforms;
2. There must always be unit tests that cover new code. New code is not
   acceptable otherwise;
3. New code must always come with extensive documentation. We have put in place
   tools to assure that code that breaks these rules can be detected where
   possible.
