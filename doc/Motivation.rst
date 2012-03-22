.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jan 11 14:43:35 2012 +0100
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
 Motivations: Be a driver to |project|!
========================================

We started developing |project| because of a common need in many research
laboratories like ours: need of a software platform where stable algorithms,
common tools, practice and start-up documentation could be put at and
maintained. It did not take long for us to understand the importance of those
tasks when involving new parties in our research: better documentation and
reliable infrastructure can boost your working efficiency many times! Since
then, we try to maintain and improve the functionality for the project with the
following ideas in mind:

* All modifications and improvements are thoroughly tested at every minor
  change: correctness comes first, whereas speed is a bonus. This is not to
  say |project| is not fast enough - it is! Our code is very much readable and
  maintanable though. We are constantly inspecting it and making sure other
  people (mainly beginners and new students) can understand what it is that our
  programs do; 
* Reproducing publications should be dead-simple. If that is not the case, your
  potential impact becomes severely compromised. We try to develop |project| so
  that our own research results are easy to reproduce and understand; 
* There should always be a laboratory-like environment in which experiments can
  be conducted without the compile-link-test cycles which tend to be lengthy
  and counter-productive. We have chosen the `Python`_ language because it
  allows a fast-paced and engaging experience. We re-use lots of available
  functionality that is already shipped with the language or are considered
  *de-facto* standards such as `NumPy`_, for efficient array manipulation and
  operation vectorization or `Matplotlib`_ for plotting.
* We still keep the ability to implement fast versions (in `C++`_) of
  identified bottlenecks. We do this by avoiding the `NumPy`_ C-API which can
  be overwhelming and bridging those types to `Blitz++`_ arrays. With that
  approach, our `C++`_ becomes clean and easy to understand. It is very easy to
  speed-up code if you need to in |project|;
* As much as is possible with a project in this scale, we try to re-use
  existing open-source packages and standards. This simplifies our testing to
  the bare minimum and keep the overall quality of |project| pretty high;
* We do this effort so |project| is useful to us (and maybe to you) as a
  continously development framework for new experiments in signal processing,
  machine learning and biometrics. We open source it (`GPL-3.0`_) with hopes it
  is useful to others. (For more information on licensing, please visit our
  :doc:`Licensing`)

History
-------

|project| has distant roots on `Torch 3 vision`_, a variant of `Torch`_, a
framework for machine learning, initially developed at the Idiap Research
Institute in Martigny, Wallis, Switzerland. It diverged from `Torch`_ in its
early infancy to become a completely independent project, focused primarily on
tools for Biometrics and Forensics, Machine Learning and Signal Processing.
Here are some links to older versions of ``Torch`` that certain served as an
inspiration:

* `Torch 3`_
* `Torch 3 Vision`_
* `Torch 5`_

Authors
-------

|project| is developed and maintained by:

.. place your name here if you participate - keep alphabetical order

* André Anjos
* Christopher McCool
* Cosmin Atanasoanei
* François Moulin
* Ivana Chingovska
* Laurent El-Shafey
* Manuel Guenther
* Murali Chakka
* Roy Wallace
* Sébastien Marcel
* Venkatesh Bala

.. include:: links.rst
