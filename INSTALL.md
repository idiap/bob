# Installing Bob

This document briefly describes how to install Bob on a generic platform. For
platform-specific details and pre-compiled binaries, please consult our website
at http://idiap.github.com/bob/.

## Dependencies

Bob depends on:

 * Blitz++
 * Boost
 * Lapack
 * giflib
 * libjpeg
 * libnetpbm
 * libpng
 * libtiff
 * HDF5

There are also optional dependencies we strongly recommend:

 * FFMpeg/LibAV
 * MatIO
 * VLFeat

Building and documentation generation depends on the following packages:

 * CMake
 * Doxygen
 * Dvipng
 * LaTeX
 * Python (>=2.4)

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
