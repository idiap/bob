# Installing Bob

This document briefly describes how to install Bob on a generic platform. For
platform-specific details and pre-compiled binaries, please consult our website
at http://idiap.github.com/bob/.

## Dependencies

Bob depends on:

 * Blitz++
 * Boost
 * Lapack
 * fftw
 * giflib
 * libjpeg
 * libnetpbm
 * libpng
 * libtiff
 * HDF5
 * Python and the following packages:
   * Setuptools
   * NumPy
   * SciPy
   * argparse
   * Sphinx
   * Nose
   * SqlAlchemy
   * Matplotlib

There are also optional dependencies we strongly recommend:

 * FFMpeg
 * MatIO
 * Qt4
 * VLFeat
 * OpenCV
 * LIBSVM

Building and documentation generation depends on the following packages:

 * CMake
 * Doxygen
 * Dvipng
 * LaTeX

## Building

Once you have built and installed all dependencies locally, you should use
CMake to build Bob itself. From your shell, do:

```sh
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```

## Testing

Tests can be executed with:

```sh
$ make test #run C++ tests
$ make nosetests #run Python tests
$ make sphinx-doctest #run Documentation tests
```

## Installing

Just execute:

```
$ make install
```

## Influential CMake variables

Some variables that may be handy:

 * CMAKE_BUILD_TYPE: options `Release` or `Debug` are supported
 * CMAKE_PREFIX_PATH: places to look-up for externals such as the dependencies
   listed above
 * WITH_PYTHON: if you would like to force a specific version of python, you
   can define it with this variable
 * BOB_INSTALL_PYTHON_INTERPRETER: installs a shell wrapper for both python and
   ipython (if you have it) that prefixes the build or installation egg
   locations
