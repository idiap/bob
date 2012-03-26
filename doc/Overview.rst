.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
.. Roy Wallace 26 Mar 2012
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

========
Overview
========

|project| is a free signal-processing and machine learning toolbox 
developed by the Biometrics group at `Idiap`_ Research Institute, Switzerland. The toolbox is
written in a mix of `Python`_ and `C++` and is designed to be both efficient
and to reduce development time. 

Below is a brief summary of what you can do right now with |project|. To get started using |project|, please continue to the instructions for :doc:`Installation`.

Mathematical and signal processing
----------------------------------

Eigenvalue decomposition, matrix inversion and other linear algebra is available and implemented using `LAPACK`_ routines at the `C++`_ level. In addition, Fast Fourier Transform is made possible via a bridge to the `FFTW`_ library.

Image processing
----------------

Numerous image processing tools are provided such as filtering (Gaussian, Median, Gabor), visual feature extraction (LBP, SIFT bridge to `VLFeat`_), face normalization and optical flow.

Machine learning
----------------

|project| has been developed by researchers tackling many machine vision problems. Several machine learning algorithms have been integrated into the library. Dimensionality reduction is supported using Principal Component Analysis, Linear Discriminant Analysis and its probabilistic variant. k-Means is provided for data clustering, and classification is possible using both generative modeling techniques (Gaussian Mixture Models, Join Factor Analysis) and discriminative approaches such as Multi-Layer Perceptrons or Support Vector Machine (via a `LIBSVM`_ bridge). 

Storing and managing data
-------------------------

The library has been designed to run on various platforms and be easily interfaced with other software. We have chosen the open and portable `HDF5`_ library and file format as our core feature for storing and managing data. `HDF5`_ is very flexible and hence allows us to store simple multi-dimensional arrays as well as complex machine learning models. Many tools for viewing, and analyzing the data are already available.  In addition, we also support the loading and storing of most image formats thanks to `ImageMagick`_, videos through `FFmpeg`_ as well as standard `MATLAB`_ file using `MatIO`_.

Database support
----------------

The library currently provides an API to easily query and interface with well know biometric databases. In particular, several protocols are integrated with the aim at improving reproducibility of scientific publications.

Performance evaluation
----------------------

A module of the library is dedicated to performance evaluation. Computation of false alarm and false rejection rate, equal error rate are supported as well as the generation of ROC, DET or EPC curves.

.. include:: links.rst
