.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 03 Nov 2014 10:37:52 CET

=====
 Bob
=====

Bob is a free signal-processing and machine learning toolbox originally
developed by the Biometrics group at `Idiap`_ Research Institute, Switzerland.

The toolbox is written in a mix of `Python`_ and `C++`_ and is designed to be
both efficient and reduce development time. It is composed of a reasonably
large number of `packages`_ that implement tools for image, audio & video
processing, machine learning and pattern recognition.

**This package is only a place-holder for Bob's** `Wiki`_ **and** `Bug
tracker`_.

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

Packages must be added using a git submodule::

  $ git submodule add https://github.com/bioidiap/bob.foo.bar.git layers/2/bob.foo.bar

Then, update the following files:

1. Add your package to the build::

   $ vim layer2.cfg

2. Add your package to the overall documentation::

   $ vim index.rst

3. Add your package to be "deploy-tested"::

   $ vim requirements2.txt

4. Add a row for your package in our Packages_ list

5. Update the dependency graph::

   $ tools/update_dependency_graphs.sh all
   
   The current dependency graph looks like that:
   
   .. image:: https://raw.githubusercontent.com/idiap/bob/master/dependencies.png
      :target: https://raw.githubusercontent.com/idiap/bob/master/dependencies.png
      :width: 50%

Updating a Package
------------------

Git submodules work by registering a precise commit hash identifier from each
submodule along with their repository locations. If you update the submodule,
you have to tell the main module that it now needs to use a new commit
identifier. If you don't do so, the main module will continue to checkout the
old version of the submodule.

In case a submodule is updated, this package will not automatically update its
reference to such database. You need to explicitly do it. To update a
submodule, first initialize this package::

  $ git submodule init
  $ git submodule update

Change to the directory where the submodule you want to update lives::

  $ cd layers/2/bob.foo.bar

The submodule repositories added by ``git submodule update`` are headless. This
means they don't have a branch. So, you must first checkout the branch with the
new revision you want to update::

  $ git checkout master
  $ git pull

Switch back to the root of the package and re-add the submodule::

  $ cd ../..
  $ git commit -m "Updated bob.foo.bar" layers/2/bob.foo.bar
  $ git push

.. External References

.. _c++: http://www2.research.att.com/~bs/C++.html
.. _python: http://www.python.org
.. _idiap: http://www.idiap.ch
.. _packages: https://github.com/idiap/bob/wiki/Packages
.. _wiki: https://github.com/idiap/bob/wiki
.. _bug tracker: https://github.com/idiap/bob/issues
.. _dependencies: https://github.com/idiap/bob/wiki/Dependencies
