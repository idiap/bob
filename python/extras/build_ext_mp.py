#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu  6 Sep 08:16:26 2012 

"""Special build_ext class that can build extensions in parallel
"""

import os
from setuptools.command.build_ext import build_ext as build_ext_base
from copy_reg import pickle
from types import MethodType

def self_headers():
  '''Discovers and catalogs all Bob headers'''

  retval = []
  for path, dirs, files in os.walk(os.path.join(PACKAGE_BASEDIR, 'include')):
    for f in files:
      if f.endswith(".h"):
        retval.append(os.path.join(path, f))
  return retval

BOB_PYTHON_HEADERS = self_headers()

class build_ext(build_ext_base):
  '''Customized extension to build bob.python bindings in the expected way'''

  def __init__(self, *args, **kwargs):
    build_ext_base.__init__(self, *args, **kwargs)

  def build_extension(self, ext):
    '''Concretely builds the extension given as input'''

    def ld_ok(opt):
      '''Tells if a certain option is a go for the linker'''

      if opt.find('-L') == 0: return False
      return True

    def calculate_dependencies(src, libs, incdir):
      '''Calculates header file dependencies'''
    
      deps = []
      for package in [k[4:] for k in libs if k.find("bob") == 0]:
        deps += BOB['headers'].get(package, [])
      deps.insert(0, os.path.join(incdir, 'bob', 'config.h'))
      deps.extend(BOB_PYTHON_HEADERS)
      return deps

    system = [k for k in ext.include_dirs if k != BOB['includedir']]
    ext.include_dirs = [k for k in ext.include_dirs if k not in system]
    ext.include_dirs.insert(0, os.path.join(PACKAGE_BASEDIR, 'include'))

    ext.depends = calculate_dependencies(ext.sources, ext.libraries, 
        BOB['includedir'])

    for k in system:
      ext.extra_compile_args.extend(('-isystem', k))

    # Some clean-up on the linker which is screwed up...
    self.compiler.linker_so = [k for k in self.compiler.linker_so if ld_ok(k)]

    build_ext_base.build_extension(self, ext)

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def worker(builder, extension):
  '''A free standing non-member method for the multiprocess pool'''

  build_ext.build_extension(builder, extension)

class build_ext_mp(build_ext):
  '''Custom extension to build bob.python bindings in parallel'''

  user_options = build_ext.user_options + [
      ('jobs=', 'j', "specify the number of parallel jobs"),
      ]

  def __init__(self, *args, **kwargs):
    build_ext.__init__(self, *args, **kwargs)

    self.registered = []

  def initialize_options(self):
    from multiprocessing import cpu_count
    self.jobs = cpu_count()/2 if not os.environ.has_key('PARALLEL_JOBS') else \
        int(os.environ['PARALLEL_JOBS'])
    build_ext.initialize_options(self)

  def finalize_options(self):
    build_ext.finalize_options(self)
    self.jobs = int(self.jobs)

  def build_extension(self, ext):
    '''Append extensions to a multiprocessing pool.

    When the pool reaches the maximum expected size, all extensions are then
    compiled in parallel processes.
    '''

    if self.jobs == 0: #normal sequential process
      build_ext.build_extension(self, ext)

    else: #do the job in parallel

      self.registered.append(ext)
      #print '[build_ext_mp] Registered extension "%s' % ext.name

      if len(self.registered) == len(EXTENSIONS):
        #print '[build_ext_mp] Building %d extensions in %d jobs' % \
        #    (len(self.registered), self.jobs)

        # remove old-style classes that don't pickle
        old_distribution = self.distribution
        self.distribution = None

        import multiprocessing
        pool = multiprocessing.Pool(self.jobs)
        res = [pool.apply_async(worker, args=(self,k)) for k in self.registered]
        pool.close()
        map(multiprocessing.pool.ApplyResult.wait, res)

        # re-stored removed elements
        self.distribution = old_distribution
        #print '[build_ext_mp] Built %d extensions in %d jobs' % \
        #    (len(self.registered), self.jobs)

pickle(MethodType, _pickle_method, _unpickle_method)

def setup_extension(ext_name, src_glob, pc_file):
  """Sets up a given C++ extension"""

  cflags = [
      '-std=c++0x',
      '-pthread',
      '-pedantic',
      '-Wno-long-long',
      '-Wno-variadic-macros',
      '-Wno-unused-function',
      ]

  import numpy

  basedir = os.path.join(PACKAGE_BASEDIR, os.path.dirname(src_glob)) 
  glob = os.path.basename(src_glob) 
  sources = [os.path.join(basedir, k) for k in fnmatch.filter(os.listdir(basedir), glob)]
  pc = pkgconfig(pc_file)

  if pc.has_key('extra_compile_args'):
    cflags += pc['extra_compile_args']

  library_dirs=pc['library_dirs'] + [BOB['boost_libdir']]

  runtime_library_dirs = None
  if BOB['soversion'].lower() == 'off':
    runtime_library_dirs = library_dirs

  return Extension(
      ext_name,
      sources=sources,
      language="c++",
      include_dirs=pc['include_dirs'] + [numpy.get_include()],
      extra_compile_args=cflags,
      library_dirs=library_dirs,
      runtime_library_dirs=runtime_library_dirs,
      libraries=pc['libraries'] + ['boost_python-mt'],
      )
