# Installing Bob

This document briefly describes how to install Bob on a generic platform. For
platform-specific details and pre-compiled binaries, please consult our website
at http://idiap.github.com/bob/.

## Dependencies

Bob depends on:

 * Blitz++
 * Boost
 * Python & NumPy
 * Lapack
 * fftw
 * ImageMagick
 * HDF5

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
$ make test
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
