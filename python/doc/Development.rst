.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 27 Mar 2012
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

.. _section-development:

=======================
 |project| Development
=======================

The core components of |project| are implemented in C++ for speed. 
Python bindings are provided to the underlying C++ code for the convenience of the user.

Before deciding to develop a new feature of |project|, we strongly recommend you follow this procedure:

1. Check carefully to see if any of the functionality you desire is already implemented in one of |project|'s :doc:`Dependencies`. 
2. If the functionality is not implemented in any of the :doc:`Dependencies`, first try to implement it in `Python`_.
3. If the `Python`_ implementation is too slow for your purposes, migrate only as much of the code as necessary into `C++`_.
4. If you have implemented functionality in `C++`_, create `Python`_ bindings as necessary, so the functionality can be used from within a `Python`_ environment (see instructions on this page: :doc:`PythonBindingsDevelopment`).

The following sections provide guidelines for |project| development in `C++`_ and `Python`_.

.. note::
  For |project| development we use the Git_ version control system. If you aren't
  familiar with Git, please refer to the official `Git tutorial`_. Within
  this guide, we assume you have already gone through all introductory material on Git
  and are ready to get your hands dirty with |project|.

 
.. toctree::
   :maxdepth: 2

   CxxDevelopment
   PythonDevelopment
   PythonBindingsDevelopment

.. include:: links.rst
.. _`Git`: http://git-scm.com/
.. _`Git Tutorial`: http://schacon.github.com/git/gittutorial.html
