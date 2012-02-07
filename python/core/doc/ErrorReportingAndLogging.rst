.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jan 11 14:43:35 2012 +0100
.. 
.. Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

=======================================
 Error Reporting and Logging in Python
=======================================

If you use Bob Python-bindings, please make use of the standard
`Python Logging Module`_ to report messages. Messages injected from the C++
code will be diverged automatically and served into the same central place, so
you can control, from python, the verbosity levels and which messages get
printed. There are two ''hats'' you can have while developing in Python. As an
application writer or as a library developer. Let's examine and give examples
for both usage scenarios.

As an application writer
------------------------

This is the most common case if you are playing with Bob as a user. As an
application writer, you just want to control the levels that are reported and
how the logs are printed. All code written in Bob should use the base `bob`
logger. Libraries written in pure python inside Bob use a child of this logger.
For example, constructions inside the `core` package should use the `bob.core`.
Please note that, as an exception, all logging messages diverged from C++ are
done using the `bob.cxx` logger. Here is the mapping adopted by the
Python-bindings from C++ messages into Python Logging levels:

* debug, all levels: `logging.DEBUG`
* info: `logging.INFO`
* warning: `logging.WARNING`
* error: `logging.ERROR`

You can configure the loggers as it pleases you. For example, here is how
to silence all messages coming from C++:

.. code-block:: python

  import bob
  import logging

  class NullHandler(logging.Handler):
    def emit(self, record):
  pass

  cxx_logger = logging.getLogger('bob.cxx')
  cxx_logger.addHandler(NullHandler())

Of course, you are free to change the logging format and other whistles as you
want for your own application.

To enable debug messages to be printed out while using the Python bindings you
have to:

* set the environment variable BOB_DEBUG to 1, 2 or 3. You can do that from
  python:

  .. code-block:: python
    
    import os
    os.environ['TORCH_DEBUG'] = '2'

* make sure the logging level is set at least to `logging.DEBUG` using the
  logger:

  .. code-block:: python

    import logging
    logging.getLogger('bob').setLevel(logging.DEBUG)

Without those precautions, no debug messages will appear.  Please note that by
default the logging module comes with `logging.WARNING` set so not even info
messages will show up unless you set it so.

As a library developer
----------------------

If you plan to develop a package, you should follow the hierarchy convention
proposed at the `logging` module documentation and inherit from the `bob` root
logger, its properties. You can accomplish this by instantiating a version of
the logger inside any file of your package (the first time it is used, it is
created automatically):

.. code-block:: python

  import logging

  # the logger bob.mypackage will inherit all configuration
  from "bob" (parent)
  # that is what you want!
  logger = logging.getLogger('bob.mypackage')

  logger.warn("Waw!")

Exception handling
------------------

We provide bindings for `bob::core::Exception` into python.  You can catch it
like this:

.. code-block:: python

  import bob

  try:
  #some bob construction
  except bob.core.Exception, e:
  print "Did not execute propertly: %s" % e

If you develop new exceptions and need them bound into python for specific
actions, please make sure to follow the recipe used to bind the C++
`bob::core::Exception` and provide good documentation.

.. place here your references:
.. _`Python Logging Module`: http://docs.python.org/library/logging.html 
