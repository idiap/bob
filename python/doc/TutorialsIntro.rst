.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

**************
 Introduction
**************

The following tutorials constitute a suitable starting point to get to know 
how to use the |project| library and to learn its fundamental concepts. 

They all rely on the |project| lab-like environment which is `Python`_. Using 
|project| within a `Python`_ environment is convenient because:

* you can easily glue together all of the components of an experiment within a
  single Python script (which does not require to be compiled),

* scripts may easily rely on other `Python`_ tools as well as |project|, and 

* `Python`_ bindings are used to transparently run the underlying efficient C++
  compiled code for the key features of the library.

.. _section-usage:

First use
---------

If you installed |project| with administrator priviledges, then using |project| is as
simple as starting `Python`_ and importing the |project| package. 
Here is an example:

.. code-block:: sh

  $ python
  ...
  >>> import bob
  >>> print(bob.version)

.. If you decided to use |project| from the build location (without
.. properly installing it) or 

If you installed without administrator priviledges, you should first check two
things:

1. Use the **same** version of `Python`_ that was used to compile |project|. 
2. Append the location of the built/installed libraries to your `PYTHONPATH`,
   so `Python`_ can find |project|.

Example:

.. code-block:: sh

  $ PYTHONPATH=<your-bob-build>/lib/python2.6 python2.6
  ...
  >>> import bob
  >>> print(bob.version)

|project| also includes some executables that are immediately usable after
installation. For example, to print the version details of your |project|
installation, just run:

.. code-block:: sh

  $ bob_config.py

In the following tutorials, we assume you have followed the above steps,
started `Python`_ and imported the |project| library as follows.

.. code-block:: python

  >>> import bob

Don't forget that you can always get information about a specific function of
|project| by using the `Python`_ help() function.

.. code-block:: python 

  >>> help(bob.machine.MLP)


.. Place here your external references

.. _python: http://www.python.org
