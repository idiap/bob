.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. Mon 20 Jul 2015 16:57:00 CEST

====================
 Bob Meta Package
====================

Bob is a free signal-processing and machine learning toolbox originally
developed by the Biometrics group at `Idiap`_ Research Institute, Switzerland.

The toolbox is written in a mix of `Python`_ and `C++`_ and is designed to be
both efficient and reduce development time. It is composed of a reasonably
large number of `packages`_ that implement tools for image, audio & video
processing, machine learning and pattern recognition.

**This is a meta package containing depencies w.r.t the layers 0 and 1 of Bob. The purpose of this package is to make it easy the deployment task in different platforms.**

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

As per-usual, make sure all external `dependencies`_ are installed on your host
before trying to compile the whole of Bob. Once all dependencies_ are
satisfied, you should be able to::

  $ python bootstrap.py
  $ ./bin/buildout

You may tweak the options in ``buildout.cfg`` to disable/enable verbosity and
debug builds, **before you run** ``./bin/buildout``.


Documentation
-------------

You can generate the documentation for all packages in this container, after
installation, using Sphinx::

  $ ./bin/sphinx-build . sphinx

This shall place in the directory ``sphinx``, the current version for the
documentation of the package.

Testing
-------

You can run a set of tests using the nose test runner::

  $ ./bin/nosetests -sv

You can run our documentation tests using sphinx itself::

  $ ./bin/sphinx-build -b doctest . sphinx

Adding a Package
----------------

   To add a package, just add it in the dependency list.


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



Updating a Package
------------------
 TODO::
 
 $ ./bin/get_versions.py
 
 

Removing a Package
------------------

   To add a package, just remove it from the dependency list.



.. External References

.. _c++: http://www2.research.att.com/~bs/C++.html
.. _python: http://www.python.org
.. _idiap: http://www.idiap.ch
.. _packages: https://github.com/idiap/bob/wiki/Packages
.. _wiki: https://github.com/idiap/bob/wiki
.. _bug tracker: https://github.com/idiap/bob/issues
.. _dependencies: https://github.com/idiap/bob/wiki/Dependencies

