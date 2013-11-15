.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 26 Mar 2012
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland


Code Base Overview
------------------

|project| code base is subdivided into packages. Each package may depend on other
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
