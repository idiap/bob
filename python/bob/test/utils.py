#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Feb  7 09:58:22 2013 

"""Re-usable decorators and utilities for Bob test code
"""

import os
import tempfile
import pkg_resources
import functools
from nose.plugins.skip import SkipTest
from distutils.version import StrictVersion as SV

def datafile(f, module=None, path='data'):
  """Returns the test file on the "data" subdirectory of the current module.

  Keyword attributes

  f: str
    This is the filename of the file you want to retrieve. Something like
    ``'movie.avi'``.

  package: module, optional
    This is the python-style package name of the module you want to retrieve
    the data from. This should be something like ``...io.test``. Note this is
    **not** a string, but the module object. If you can reach it already, you
    must import it first.

  path: str, optional
    This is the subdirectory where the datafile will be taken from inside the
    module. Normally (the default) ``data``. It can be set to ``None`` if it
    should be taken from the module path root (where the ``__init__.py`` file
    sits).
 
  Returns the full path of the file.
  """
  
  resource = __name__ if module is None else module.__name__
  final_path = f if path is None else os.path.join(path, f)
  return pkg_resources.resource_filename(resource, final_path)

def temporary_filename(prefix='bobtest_', suffix='.hdf5'):
  """Generates a temporary filename to be used in tests"""

  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

# Here is a table of ffmpeg versions against libavcodec, libavformat and
# libavutil versions
ffmpeg_versions = {
    '0.5':  [ SV('52.20.0'),   SV('52.31.0'),   SV('49.15.0')   ],
    '0.6':  [ SV('52.72.2'),   SV('52.64.2'),   SV('50.15.1')   ],
    '0.7':  [ SV('52.122.0'),  SV('52.110.0'),  SV('50.43.0')   ],
    '0.8':  [ SV('53.7.0'),    SV('53.4.0'),    SV('51.9.1')    ],
    '0.9':  [ SV('53.42.0'),   SV('53.24.0'),   SV('51.32.0')   ],
    '0.10': [ SV('53.60.100'), SV('53.31.100'), SV('51.34.101') ],
    '0.11': [ SV('54.23.100'), SV('54.6.100'),  SV('51.54.100') ],
    '1.0':  [ SV('54.59.100'), SV('54.29.104'), SV('51.73.101') ],
    '1.1':  [ SV('54.86.100'), SV('54.59.106'), SV('52.13.100') ],
    }

def ffmpeg_found(version_geq=None):
  '''Decorator to check if a codec is available before enabling a test
  
  To use this, decorate your test routine with something like:

  .. code-block:: python

    @ffmpeg_found()

  You can pass an optional string to require that the FFMpeg version installed
  is greater or equal that version identifier. For example:

  .. code-block:: python

    @ffmpeg_found('0.10') #requires at least version 0.10

  Versions you can test for are set in the ``ffmpeg_versions`` dictionary in
  this module.
  '''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      try:
        from ..io._io import version
        avcodec_inst= SV(version['FFmpeg']['avcodec'])
        avformat_inst= SV(version['FFmpeg']['avformat'])
        avutil_inst= SV(version['FFmpeg']['avutil'])
        if version_geq is not None:
          avcodec_req,avformat_req,avutil_req = ffmpeg_versions[version_geq]
          if avcodec_inst < avcodec_req:
            raise SkipTest('FFMpeg/libav version installed (%s) is smaller than required for this test (%s)' % (version['FFmpeg']['ffmpeg'], version_geq))
        return test(*args, **kwargs)
      except KeyError:
        raise SkipTest('FFMpeg was not available at compile time')

    return wrapper

  return test_wrapper

def codec_available(codec):
  '''Decorator to check if a codec is available before enabling a test'''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      from ..io import supported_video_codecs
      d = supported_video_codecs()
      if d.has_key(codec) and d[codec]['encode'] and d[codec]['decode']:
        return test(*args, **kwargs)
      else:
        raise SkipTest('A functional codec for "%s" is not installed with FFmpeg' % codec)

    return wrapper

  return test_wrapper

def visioner_available(test):

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from .. import visioner
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest("The visioner module is not available")

  return wrapper

def libsvm_available(test):

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      from ..machine import SupportVector
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest("The visioner module is not available")

  return wrapper
