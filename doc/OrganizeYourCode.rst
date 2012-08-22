.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Wed 15 Aug 09:08:47 2012 

==========================================
 Organize Your Work in Satellite Packages
==========================================

This tutorial explains how to use `zc.buildout <http://www.buildout.org/>`_ to
build complete `Python`-based working environments (a.k.a `packages`) which are
isolated from your current python installation. By following this recipe you
will be able to:

* Create a basic working environment using either a stock |project|
  installation or your own compiled (and possibly uninstalled) version of
  |project|;
* Install additional python packages to augment your virtual work environment
  capabilities - e.g., to include a new python package for a specific purpose
  covering functionality that does not necessarily exists in |project|;
* Distribute your work to others in a clean and organized manner.

The advantaged of using ``zc.buildout`` is that it does **not** require
administrator privileges for setting up all of the above. Furthermore, you will
be able to create **isolated** environments for each project you have,
individually. This is a great way, for example, to release code for laboratory
exercises or for a particular publication that depends on |project|.

.. note::
  The core of our strategy is based on standard tools for *defining* and
  *deploying* Python packages. If you are not familiar with Python's
  ``setuptools``, ``distutils`` or PyPI, it can be beneficial to `learn about
  those <http://packages.python.org/an_example_pypi_project/setuptools.html>`_
  before you start. Python `Setuptools
  <http://pypi.python.org/pypi/setuptools/>`_ and `Distutils
  <http://docs.python.org/distutils/>`_ are mechanisms to *define and
  distribute* python code in a packaged format, optionally through `PyPI
  <http://pypi.python.org/pypi>`_, a web-based Python package index and
  distribution portal. 
  
  `Buildout <http://www.buildout.org>`_ is a tool to *deploy* Python packages
  locally, automatically setting up and isolating your environment.

Anatomy of a buildout Python package
------------------------------------

The best way to create your package is to download a `skeleton from the Idiap
github website <https://github.com/idiap/bob.project.example>`_ and build on
it, modifying what you need. Fire-up a shell window and than do this:

.. code-block:: sh

  $ git clone --depth=1 https://github.com/idiap/bob.project.example.git
  $ cd bob.project.example
  $ rm -rf .git #this is optional - you won't need the .git directory

We now recommend you read the file ``readme.rst`` situated at the root of the
just downloaded material. It contains important information on other
functionality such as document generation and unit testing, which will not be
covered on this introductory material.

The anatomy of a minimal package should look like the following:

.. code-block:: sh

  .
  |-- LICENSE.txt      # a text file with licensing information
  |-- MANIFEST.in      # describes which files should be installed, besides the python ones
  |-- README.rst       # a descriptive explanation of the package contents, in restructured-text format
  |-- bootstrap.py     # stock script downloaded from http://svn.zope.org/*checkout*/zc.buildout/trunk/bootstrap/bootstrap.py
  |-- buildout.cfg     # buildout configuration to create a local working environment for this package
  |-- foo              # the package 'foo'
  |   |-- __init__.py  # every package should have __init__.py file on its root
  |   |-- bar.py       # a module inside the package 'foo'
  |-- setup.py         # installation + requirements for this particular package

Our example that you just downloaded contains these files and a few extra ones
useful for this tutorial. Inspect the package so you are aware of its contents.
All files are in text format and should be heavily commented. The most
important file that requires your attention is ``setup.py``. This file contains
the basic information for the Python package you will be creating and defines
scripts it creates and dependencies it requires for execution. To customize the
package to your needs, you will need to edit this file and modify it
accordingly. Before doing so, it is suggested you go through all of this
tutorial so you are familiar with the whole environment. The example package,
as it is distributed, contains a fully working example.

In the remaining of this text, we start by explaining the simplest use case:
when you have |project| centrally installed so you can import it on your Python
prompt without further setup. If that is not the case, you will need to follow
the supplementary instructions down below so to make ``buildout`` aware of
|project|'s installation location. Both are easy, but let's start with the
simplest one.

You have |project| centrally installed
======================================

This is the typical case when you have installed one of our `pre-packaged
versions of Bob <https://github.com/idiap/bob/wiki/Packages>`_ or you have
setup your account or machine so that |project| is automatically found when you
start your Python prompt. To check if you satisfy that condition, just fire up
Python and try to ``import bob``:

.. code-block:: sh

  $ python
  >>> import bob

If that works, setting-up your work environment is no different than what is
described on the ``buildout`` website. `Here is a screencast
<http://video.google.com/videoplay?docid=3428163188647461098&hl=en>`_ by the
author of ``buildout`` that explains that process in details.

The package you cloned above contains all elements to get you started. It
defines a single library inside called ``example``, which declares a simple
script, called ``version.py`` that prints out the version of |project|. When you
clone the package, you will not find any executable as ``buildout`` needs to
check all dependencies and install missing ones before you can execute
anything. Here is how to go from nothing to everything:

.. code-block:: sh

  $ python bootstrap.py
    Downloading http://pypi.python.org/packages/2.6/s/setuptools/setuptools-0.6c11-py2.6.egg
    Creating directory '/Users/andre/Projects/bob.project.example/bin'.
    Creating directory '/Users/andre/Projects/bob.project.example/parts'.
    Creating directory '/Users/andre/Projects/bob.project.example/eggs'.
    Creating directory '/Users/andre/Projects/bob.project.example/develop-eggs'.
    Getting distribution for 'setuptools'.
    Got setuptools 0.6c12dev-r88846.
    Generated script '/Users/andre/Projects/bob.project.example/bin/buildout'.
  $ ./bin/buildout
    Develop: '/Users/andre/Projects/bob.project.example/.'
    Installing python.
    Generated script '/Users/andre/Projects/bob.project.example/bin/version.py'.
    Generated interpreter '/Users/andre/Projects/bob.project.example/bin/python'.

.. note::

  The python shell used in the first line of the previous command set
  determines the python interpreter that will be used for all scripts developed
  inside this package. Because this package makes use of Bob, you must make
  sure that the bootstrap.py script is called with the same interpreter used to
  build Bob, or unexpected problems might occur.

  If Bob is installed by the administrator of your system, it is safe to
  consider it uses the default python interpreter. In this case, the above 2
  command lines should work as expected.

You should now be able to execute ``./bin/version.py``:

.. code-block:: sh

  $ ./bin/version.py 
    The installed version of Bob is '1.0.2'

Everything is now setup for you to continue the development of this package.
Modify all required files to setup your own package name, description and
dependencies. Start adding files to your library (or libraries) and, if you
wish, make this package available in a place with public access to make your
research public. We recommend using Github. Optionally, `drop-us a
message <https://groups.google.com/forum/?fromgroups#!forum/bob-devel>`_
talking about the availability of this package so we can add it to the `growing
list of available software
<https://github.com/idiap/bob/wiki/Satellite-Packages>`_.

|project| is installed somewhere else
=====================================

This is the typical case when you compile |project| from scratch, yourself, and
decided not to install it formally in some automatically scanned location (like
``/usr``). For example, you may want to test a new version of |project| with
your setup or check which API changes will affect your released code. In such
cases, you will need to tell ``buildout`` what is the base build directory
**or** installation prefix for |project|.

The example package you downloaded contains an alternative buildout
configuration file called ``localbob.cfg``. Edit this file by locating the
section named ``bob`` inside it and editing the value of the variable
``install-directory`` to point to the root of your build or installation
directory for |project|. Any of the two will work ok. That modification should
do all the magic. Proceed normally with the boostrap and ``buildout`` phases
using the ``localbob.cfg`` configuration:

.. code-block:: sh

  $ python bootstrap.py
  $ ./bin/buildout -c localbob.cfg

After buildout has run, you should be able to execute ``bin/version.py`` again.

.. note::

  The python shell used in the first line of the previous command set
  determines the python interpreter that will be used for all scripts developed
  inside this package. Because this package makes use of Bob, you must make
  sure that the bootstrap.py script is called with the same interpreter used to
  build Bob, or unexpected problems might occur.

  If Bob is installed by the administrator of your system, it is safe to
  consider it uses the default python interpreter. In this case, the above 2
  command lines should work as expected.

Document Generation and Unit Testing
====================================

In the file ``readme.rst`` you will find important information explaining how
to generate documentation and run unittests for your satellite package. If you
intend to distribute this material, please consider carefully documenting and
creating unit tests for your package. Documentation is a great starting point
for users and unit tests can be used to check funcionality in unexpected
circumstances such as variations in package versions.

Creating Database Satellite Packages
------------------------------------

Database satellite packages are special satellite packages that can hook-in
|project|'s database manager ``bob_dbmanage.py``. Except for this detail, they
should look exactly like a normal package.

To allow the database to be hooked to the ``bob_dbmanage.py`` you must
implement a non-virtual python class that inherits from
:py:class:`bob.db.driver.Interface`. Your concrete implementation should then
be described at the ``setup.py`` file with a special ``bob.db`` entry point:

.. code-block:: python

    # bob database declaration
    'bob.db': [
      'replay = replay.db.driver:Interface',
      ],

At present, there is no formal design guide for databases. Nevertheless, it is
considered a good practice to follow the design of `currently existing database
packages <https://github.com/idiap/bob/wiki/Satellite-Packages>`_. This should
ease migration in case of future changes.

Satellite Packages Available
----------------------------

Look `here for our growing list of Satellite Packages <https://github.com/idiap/bob/wiki/Satellite-Packages>`_.
