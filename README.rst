.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. Mon 20 Jul 2015 16:57:00 CEST


.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/idiap/bob/master/index.html
.. image:: https://travis-ci.org/bioidiap/bob.learn.em.svg?branch=v2.3.4
   :target: https://travis-ci.org/idiap/bob?branch=v2.3.4
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/idiap/bob/tree/master
.. image:: http://img.shields.io/pypi/v/bob.png
   :target: https://pypi.python.org/pypi/bob
.. image:: http://img.shields.io/pypi/dm/bob.png
   :target: https://pypi.python.org/pypi/bob

====================
 Bob
====================

Bob is a free signal-processing and machine learning toolbox originally
developed by the Biometrics group at `Idiap`_ Research Institute, Switzerland.

The toolbox is written in a mix of `Python`_ and `C++`_ and is designed to be
both efficient and reduce development time. It is composed of a reasonably
large number of `packages`_ that implement tools for image, audio & video
processing, machine learning and pattern recognition.

If just want to use Bob's functionalities on your experiments, you are **not**
supposed to install this package on your machine, but rather `create your own
personalised work environment
<https://github.com/idiap/bob/wiki/Installation>`_ depending on your needs, by
collecting individual sub-packages based on your requirements.

If you are developing Bob packages which are supposed to built along side our
`nightly build system <https://www.idiap.ch/software/bob/buildbot/waterfall>`_,
please read on.


Installation
------------

.. note::
  If you are reading this page through our GitHub portal and not through PyPI,
  note **the development tip of the package may not be stable** or become
  unstable in a matter of moments.

  Go to `http://pypi.python.org/pypi/bob
  <http://pypi.python.org/pypi/bob>`_ to download the latest
  stable version of this package.

There are 2 options you can follow to get this package installed and
operational on your computer: you can use automatic installers like `pip
<http://pypi.python.org/pypi/pip/>`_ (or `easy_install
<http://pypi.python.org/pypi/setuptools>`_) or manually download, unpack and
use `zc.buildout <http://pypi.python.org/pypi/zc.buildout>`_ to create a
virtual work environment just for this package.

Using an automatic installer
============================

Using ``pip`` is the easiest (shell commands are marked with a ``$`` signal)::

  $ pip install bob

You can also do the same with ``easy_install``::

  $ easy_install bob

This will download and install this package plus any other required
dependencies. It will also verify if the version of Bob you have installed
is compatible.

This scheme works well with virtual environments by `virtualenv
<http://pypi.python.org/pypi/virtualenv>`_ or if you have root access to your
machine. Otherwise, we recommend you use the next option.

Using ``zc.buildout``
=====================

Download the latest version of this package from `PyPI
<http://pypi.python.org/pypi/bob>`_ and unpack it in your
working area. The installation of the toolkit itself uses `buildout
<http://www.buildout.org/>`_. You don't need to understand its inner workings
to use this package. Here is a recipe to get you started::
  
  $ python bootstrap-buildout.py
  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and
get you a fully operational test and development environment.

.. note::
  As per-usual, make sure all external `dependencies`_ are installed on your host
  before trying to compile the whole of Bob.


Documentation
-------------

You can generate the documentation for all packages in this container, after
installation, using Sphinx::

  $ ./bin/sphinx-build . sphinx

This shall place in the directory ``sphinx``, the current version for the
documentation of the package.


For the maintainers
-------------------

In the next subsections we have instructions for the maintainers of the package.

Adding a dependency package
===========================

   
   To add a package on bob, just append the package name in the file ('requirements.txt').

.. warning::
   Before adding a package to this prototype, please ensure that the package:

   * contains a README clearly indicating how to install the package (including
     external dependencies required). Also, please add package badges for the
     build status and coverage as shown in other packages (even if your package
     is not yet integrated to Travis or Coveralls).

   * Has unit tests.

   * Is integrated with Travis-CI, and correctly tests on that platform (i.e.
     it builds, it tests fine and a documentation can be constructed and tested
     w/o errors)

   * Is integrated with Coveralls for reporting test coverage

   If you don't know how to do this, ask for information on the bob-devel
   mailing list.


Updating the dependencies
=========================

 If you want to update the version of the dependency packages, run the following commands::
 
 $ ./bin/get_versions.py > requirements.txt
 $ git commit requirements.txt -m "Update requeriments" && git push
 

Removing a dependency package
=============================

   To remove a package on bob, just append the package name in the file ('requirements.txt').


.. External References

.. _c++: http://www2.research.att.com/~bs/C++.html
.. _python: http://www.python.org
.. _idiap: http://www.idiap.ch
.. _packages: https://github.com/idiap/bob/wiki/Packages
.. _wiki: https://github.com/idiap/bob/wiki
.. _bug tracker: https://github.com/idiap/bob/issues
.. _dependencies: https://github.com/idiap/bob/wiki/Dependencies

