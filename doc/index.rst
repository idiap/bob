.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 17 Feb 2014 17:40:07 CET
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

=======================
 Bob
=======================

Bob is a free signal-processing and machine learning toolbox originally
developed by the Biometrics group at `Idiap`_ Research Institute, Switzerland.

The toolbox is written in a mix of `Python`_ and `C++`_ and is designed to be
both efficient and reduce development time. It is composed of a reasonably
large number of `packages`_ that implement tools for image, audio & video
processing, machine learning and pattern recognition.

Bob is organized in several independent python packages.
Below you can find the list of all packages organized by category with their respectives documentation and building status.

.. todolist::

Basic Functionality
-------------------
   
   
.. raw:: html   

  <table>
     <tr>
       <td><a href="https://pypi.python.org/pypi/bob.core">bob.core</a></td>
       <td>
         <a href="http://pythonhosted.org/bob.core/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
         <a href="https://travis-ci.org/bioidiap/bob.core"><img src="https://travis-ci.org/bioidiap/bob.core.png?branch=master"/></a>
         <a href='https://coveralls.io/r/bioidiap/bob.core'><img src='https://coveralls.io/repos/bioidiap/bob.core/badge.png' alt='Coverage Status' /></a>
       </td>
     </tr>

     <tr>
       <td><a href="https://pypi.python.org/pypi/bob.math">bob.math</a></td>
       <td>
         <a href="http://pythonhosted.org/bob.math/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
         <a href="https://travis-ci.org/bioidiap/bob.math"><img src="https://travis-ci.org/bioidiap/bob.math.png?branch=master"/></a>
         <a href='https://coveralls.io/r/bioidiap/bob.math'><img src='https://coveralls.io/repos/bioidiap/bob.math/badge.png' alt='Coverage Status' /></a>
       </td>
     </tr>

     <tr>
       <td><a href="https://pypi.python.org/pypi/bob.measure">bob.measure</a></td>
       <td>
         <a href="http://pythonhosted.org/bob.measure/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
         <a href="https://travis-ci.org/bioidiap/bob.measure"><img src="https://travis-ci.org/bioidiap/bob.measure.png?branch=master"/></a>
         <a href='https://coveralls.io/r/bioidiap/bob.measure'><img src='https://coveralls.io/repos/bioidiap/bob.measure/badge.png' alt='Coverage Status' /></a>
       </td>
     </tr>
   </table>


Data Input and Output
---------------------
.. raw:: html   

  <table>
   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.io.base">bob.io.base</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.io.base/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.io.base"><img src="https://travis-ci.org/bioidiap/bob.io.base.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.io.base'><img src='https://coveralls.io/repos/bioidiap/bob.io.base/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.io.image">bob.io.image</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.io.image/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.io.image"><img src="https://travis-ci.org/bioidiap/bob.io.image.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.io.image'><img src='https://coveralls.io/repos/bioidiap/bob.io.image/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.io.video">bob.io.video</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.io.video/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.io.video"><img src="https://travis-ci.org/bioidiap/bob.io.video.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.io.video'><img src='https://coveralls.io/repos/bioidiap/bob.io.video/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.io.matlab">bob.io.matlab</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.io.matlab/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.io.matlab"><img src="https://travis-ci.org/bioidiap/bob.io.matlab.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.io.matlab'><img src='https://coveralls.io/repos/bioidiap/bob.io.matlab/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>
  </table>


Signal, Audio, Image and Video Processing
-----------------------------------------
.. raw:: html

  <table>
   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.sp">bob.sp</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.sp/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.sp"><img src="https://travis-ci.org/bioidiap/bob.sp.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.sp'><img src='https://coveralls.io/repos/bioidiap/bob.sp/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.ap">bob.ap</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.ap/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.ap"><img src="https://travis-ci.org/bioidiap/bob.ap.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.ap'><img src='https://coveralls.io/repos/bioidiap/bob.ap/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.ip.base">bob.ip.base</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.ip.base/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.ip.base"><img src="https://travis-ci.org/bioidiap/bob.ip.base.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.ip.base'><img src='https://coveralls.io/repos/bioidiap/bob.ip.base/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.ip.color">bob.ip.color</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.ip.color/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.ip.color"><img src="https://travis-ci.org/bioidiap/bob.ip.color.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.ip.color'><img src='https://coveralls.io/repos/bioidiap/bob.ip.color/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.ip.draw">bob.ip.draw</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.ip.draw/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.ip.draw"><img src="https://travis-ci.org/bioidiap/bob.ip.draw.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.ip.draw'><img src='https://coveralls.io/repos/bioidiap/bob.ip.draw/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>


   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.ip.gabor">bob.ip.gabor</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.ip.gabor/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.ip.gabor"><img src="https://travis-ci.org/bioidiap/bob.ip.gabor.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.ip.gabor'><img src='https://coveralls.io/repos/bioidiap/bob.ip.gabor/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>
  </table>


Machine Learning
----------------
.. raw:: html

  <table>
   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.learn.linear">bob.learn.linear</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.learn.linear/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.learn.linear"><img src="https://travis-ci.org/bioidiap/bob.learn.linear.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.learn.linear'><img src='https://coveralls.io/repos/bioidiap/bob.learn.linear/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.learn.mlp">bob.learn.mlp</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.learn.mlp/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.learn.mlp"><img src="https://travis-ci.org/bioidiap/bob.learn.mlp.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.learn.mlp'><img src='https://coveralls.io/repos/bioidiap/bob.learn.mlp/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.learn.activation">bob.learn.activation</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.learn.activation/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.learn.activation"><img src="https://travis-ci.org/bioidiap/bob.learn.activation.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.learn.activation'><img src='https://coveralls.io/repos/bioidiap/bob.learn.activation/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.learn.libsvm">bob.learn.libsvm</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.learn.libsvm/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.learn.libsvm"><img src="https://travis-ci.org/bioidiap/bob.learn.libsvm.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.learn.libsvm'><img src='https://coveralls.io/repos/bioidiap/bob.learn.libsvm/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.learn.em">bob.learn.em</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.learn.em/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.learn.em"><img src="https://travis-ci.org/bioidiap/bob.learn.em.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.learn.em'><img src='https://coveralls.io/repos/bioidiap/bob.learn.em/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>

   <tr>
     <td><a href="https://pypi.python.org/pypi/bob.learn.boosting">bob.learn.boosting</a></td>
     <td>
       <a href="http://pythonhosted.org/bob.learn.boosting/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
       <a href="https://travis-ci.org/bioidiap/bob.learn.boosting"><img src="https://travis-ci.org/bioidiap/bob.learn.boosting.png?branch=master"/></a>
       <a href='https://coveralls.io/r/bioidiap/bob.learn.boosting'><img src='https://coveralls.io/repos/bioidiap/bob.learn.boosting/badge.png' alt='Coverage Status' /></a>
     </td>
   </tr>
  </table>


Modules for Developers
----------------------

.. raw:: html

  <table>
     <tr>
       <td><a href="https://pypi.python.org/pypi/bob.buildout">bob.buildout</a></td>
       <td>
         <a href="http://pythonhosted.org/bob.buildout/index.html"><img src="http://img.shields.io/badge/docs-missing-red.png"/></a>
         <a href="https://travis-ci.org/bioidiap/bob.buildout"><img src="https://travis-ci.org/bioidiap/bob.buildout.png?branch=master"/></a>
         <a href='https://coveralls.io/r/bioidiap/bob.buildout'><img src='https://coveralls.io/repos/bioidiap/bob.buildout/badge.png' alt='Coverage Status' /></a>
       </td>
     </tr>

     <tr>
       <td><a href="https://pypi.python.org/pypi/bob.extension">bob.extension</a></td>
       <td>
         <a href="http://pythonhosted.org/bob.extension/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
         <a href="https://travis-ci.org/bioidiap/bob.extension"><img src="https://travis-ci.org/bioidiap/bob.extension.png?branch=master"/></a>
         <a href='https://coveralls.io/r/bioidiap/bob.extension'><img src='https://coveralls.io/repos/bioidiap/bob.extension/badge.png' alt='Coverage Status' /></a>
       </td>
     </tr>

     <tr>
       <td><a href="https://pypi.python.org/pypi/bob.blitz">bob.blitz</a></td>
       <td>
         <a href="http://pythonhosted.org/bob.blitz/index.html"><img src="http://img.shields.io/badge/docs-stable-yellow.png"/></a>
         <a href="https://travis-ci.org/bioidiap/bob.blitz"><img src="https://travis-ci.org/bioidiap/bob.blitz.png?branch=master"/></a>
         <a href='https://coveralls.io/r/bioidiap/bob.blitz'><img src='https://coveralls.io/repos/bioidiap/bob.blitz/badge.png' alt='Coverage Status' /></a>
       </td>
     </tr>

   </table>

.. _c++: http://www2.research.att.com/~bs/C++.html
.. _python: http://www.python.org
.. _idiap: http://www.idiap.ch
.. _packages: https://github.com/idiap/bob/wiki/Packages

