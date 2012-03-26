.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 26 Mar 2012
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


Code Base Overview
------------------

|project| code base is subdivided in packages. Each package may depend on other
packages to work properly or on external software. There is no notion of
layering in the software structure. |project| is actually composed of a number
of re-usable components that can be deployed either separately or jointly
depending on user requirements. The following diagram may help you understand
what is the (loose) inter-dependency of |project|'s internal packages and
external software. Optional packages and external dependencies are marked with
dashed lines. Functionality shipped with the build you are currently using will
depend on software availability during compilation.

.. only:: not latex

  .. figure:: img/overview.png
    :alt: Overview of |project| packages and organization
    :scale: 50%

    Organization of |project| packages and main dependencies.

.. only:: latex

  .. figure:: img/overview.pdf
    :alt: Overview of |project| packages and organization
    :scale: 80%

    Organization of |project| packages and main dependencies.

.. include:: links.rst
