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

.. _section-installation:

======================
 Installing |project|
======================

This section describes how to automatically install pre-packaged versions of |project|.
This is the easiest way to start using |project| in your scripts, all you need to do is follow
one of the available installation methods below. On the other hand, if they are not appropriate 
or you would like to become a developer of |project|, you should compile |project| yourself as 
explained at :doc:`Compilation`. 


Ubuntu
------

We provide `Ubuntu packages`_ through its launchpad platform. To install the
packages you will need administration rights to your machine. Then:

.. code-block:: sh

  $ sudo add-apt-repository ppa:biometrics/bob
  $ sudo apt-get update
  $ sudo apt-get install bob

We recommend you keep accessing `our website`_ for the manuals, but if you
would like to keep them close, just install the ``bob-doc`` package:

.. code-block:: sh

  $ sudo apt-get install bob-doc

If you intend to develop your own extensions that are based on |project|, you
should also install the development files:

.. code-block:: sh

  $ sudo apt-get install bob-dev

Mac OSX
-------

We provide a Portfile repository that contains the description of |project|. To
use it, you first have to install `MacPorts`_ as explained on their home page.
Then, edit your ``/opt/local/etc/macports/sources.conf`` file to contain our
ports repository ``rsync://www.idiap.ch/software/bob/ports`` by inserting an
entry **before** the last entry in that file like this:

.. code-block:: sh

  rsync://www.idiap.ch/software/bob/ports/

Once that is done, installing |project| should be straight forward:

.. code-block:: sh

  $ sudo port selfupdate
  $ sudo port install bob +python26 +docs

.. Place here references to all citations in lower case

.. _ubuntu packages: https://launchpad.net/~biometrics/+archive/bob
.. _macports: http://www.macports.org/install.php
.. _our website: http://idiap.github.com/bob/

