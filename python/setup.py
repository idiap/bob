#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 16 Aug 2012 11:36:19 CEST

"""A setup file for Bob Python bindings using Boost.Python
"""

import os
import sys
from setuptools.command.build_ext import build_ext as build_ext_base
from setuptools import Extension
import subprocess

PACKAGE_BASEDIR = os.path.dirname(os.path.abspath(__file__))

# If Python < 2.7 or 3.0 <= Python < 3.2, require some more stuff
EXTRA_REQUIREMENTS = []
if sys.version_info[:2] < (2, 7) or ((3,0) <= sys.version_info[:2] < (3,2)):
  EXTRA_REQUIREMENTS.append('argparse')

# Installing in a caged environment
DESTDIR = os.environ.get('DESTDIR', '')

# ---------------------------------------------------------------------------#
#  various functions and classes to help on the setup                        #
# ---------------------------------------------------------------------------#

def pkgconfig(package):

  def uniq(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

  flag_map = {
      '-I': 'include_dirs',
      '-L': 'library_dirs',
      '-l': 'libraries',
      }

  cmd = [
      'pkg-config',
      '--libs',
      '--cflags',
      package,
      ]

  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT)

  output = proc.communicate()[0]

  if proc.returncode != 0: return {}

  kw = {}

  for token in output.split():
    if flag_map.has_key(token[:2]):
      kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])

    else: # throw others to extra_link_args
      kw.setdefault('extra_compile_args', []).append(token)

  for k, v in kw.iteritems(): # remove duplicated
      kw[k] = uniq(v)

  return kw

def bob_variables():

  def get_var(name):
    cmd = [
        'pkg-config',
        '--variable=%s' % name,
        'bob',
        ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    var = proc.communicate()[0].strip()
    if proc.returncode != 0: return None
    return var


  cmd = [
      'pkg-config',
      '--modversion',
      'bob',
      ]

  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT)

  output = proc.communicate()[0].strip()

  kw = {}
  kw['version'] = output if proc.returncode == 0 else None

  if kw['version'] is None:
    raise RuntimeError, 'Cannot retrieve Bob version from pkg-config:\n%s' % \
        output

  kw['soversion'] = get_var('soversion')

  kw['base_libdir'] = get_var('libdir')
  kw['base_includedir'] = get_var('includedir')

  return kw

# Retrieve central, global variables from Bob's C++ build
BOB = bob_variables()

class build_ext(build_ext_base):
  '''Customized extension to build bob.python bindings in the expected way'''

  linker_is_smart = None

  def __init__(self, *args, **kwargs):
    build_ext_base.__init__(self, *args, **kwargs)

  def build_extension(self, ext):
    '''Concretely builds the extension given as input'''

    def linker_can_remove_symbols(linker):
      '''Tests if the `ld` linker can remove unused symbols from linked
      libraries. In this case, use the --no-as-needed flag during link'''

      import tempfile
      f, name = tempfile.mkstemp()
      del f

      cmd = linker + ['-Wl,--no-as-needed', '-lm', '-o', name]
      proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT)
      output = proc.communicate()[0]
      if os.path.exists(name): os.unlink(name)
      return True if proc.returncode == 0 else False

    def ld_ok(opt):
      '''Tells if a certain option is a go for the linker'''

      if opt.find('-L') == 0: return False
      return True

    # Some clean-up on the linker which is screwed up...
    self.compiler.linker_so = [k for k in self.compiler.linker_so if ld_ok(k)]

    if self.linker_is_smart is None:
      self.linker_is_smart = linker_can_remove_symbols(self.compiler.linker_so)
      if self.linker_is_smart: self.compiler.linker_so += ['-Wl,--no-as-needed']

    build_ext_base.build_extension(self, ext)

def setup_extension(ext_name, pc_file):
  """Sets up a given C++ extension"""

  import numpy

  pc = pkgconfig(pc_file + '%d%d' % sys.version_info[:2])

  library_dirs=pc.get('library_dirs', [])
  library_dirs=[k for k in library_dirs if os.path.exists(k)]
  include_dirs=pc.get('include_dirs', [])
  include_dirs=[k for k in include_dirs if os.path.exists(k)]

  if len(DESTDIR) != 0:
    # Treat special caged builds
    library_dirs.insert(0, os.path.join(DESTDIR, BOB['base_libdir'].lstrip(os.sep)))
    include_dirs.insert(0, os.path.join(DESTDIR, BOB['base_includedir'].strip(os.sep)))

  runtime_library_dirs = None
  if BOB['soversion'].lower() == 'off':
    runtime_library_dirs = library_dirs

  return Extension(
      ext_name,
      sources=[],
      language="c++",
      include_dirs=include_dirs + [numpy.get_include()],
      library_dirs=library_dirs,
      runtime_library_dirs=runtime_library_dirs,
      libraries=pc['libraries'],
      )

# ---------------------------------------------------------------------------#
#  setup variables, modules and extra files declarations                     #
# ---------------------------------------------------------------------------#

CONSOLE_SCRIPTS = [
  'bob_config.py = bob.script.config:main',
  'bob_dbmanage.py = bob.db.script.dbmanage:main',
  'bob_compute_perf.py = bob.measure.script.compute_perf:main',
  'bob_eval_threshold.py = bob.measure.script.eval_threshold:main',
  'bob_apply_threshold.py = bob.measure.script.apply_threshold:main',
  'bob_plot_cmc.py = bob.measure.script.plot_cmc:main',
  'bob_face_detect.py = bob.visioner.script.facebox:main',
  'bob_face_keypoints.py = bob.visioner.script.facepoints:main',
  'bob_visioner_trainer.py = bob.visioner.script.trainer:main',
  ]

# built-in databases
DATABASES = [
    'iris = bob.db.iris.driver:Interface',
    ]

# test data that needs to be shipped with the distribution
def find_all_test_data():

  def add_data(l, path):
    remove = os.path.join(PACKAGE_BASEDIR, 'bob') + os.sep
    for sub_path, dirs, files in os.walk(path):
      for f in files:
        path = os.path.join(sub_path, f).replace(remove, '')
        l.append(path)

  retval = []

  package_dir = os.path.join(PACKAGE_BASEDIR, 'bob')
  for pack in os.listdir(package_dir):
    sub_package_dir = os.path.join(package_dir, pack)
    if not os.path.isdir(sub_package_dir): continue
    for subdir in os.listdir(sub_package_dir):
      if subdir == 'test':
        test_path = os.path.join(PACKAGE_BASEDIR, 'bob', pack, 'test', 'data')
        if os.path.exists(test_path):
          add_data(retval, test_path)

  return retval

DATAFILES = find_all_test_data()

# hand-picked data files to be shipped with the distribution
DATAFILES += [
    'db/iris/iris.names',
    'db/iris/iris.data',
    ]

EXTENSIONS = [
    setup_extension('bob.core._core', 'bob-core-py'),
    setup_extension('bob.core.random._core_random', 'bob-core-random-py'),
    setup_extension('bob.io._io', 'bob-io-py'),
    setup_extension('bob.math._math', 'bob-math-py'),
    setup_extension('bob.measure._measure', 'bob-measure-py'),
    setup_extension('bob.sp._sp', 'bob-sp-py'),
    setup_extension('bob.ip._ip', 'bob-ip-py'),
    setup_extension('bob.ap._ap', 'bob-ap-py'),
    setup_extension('bob.machine._machine', 'bob-machine-py'),
    setup_extension('bob.trainer._trainer', 'bob-trainer-py'),
    setup_extension('bob.trainer.overload._trainer_overload', 'bob-trainer-overload-py'),
    ]

if pkgconfig('bob-daq'):
  EXTENSIONS.append(
      setup_extension('bob.daq._daq', 'bob-daq-py')
      )

if pkgconfig('bob-visioner'):
  EXTENSIONS.append(
    setup_extension('bob.visioner._visioner', 'bob-visioner-py')
    )

  DATAFILES += [
      'visioner/detection.gz',
      'visioner/localization.gz',
      ]

# ---------------------------------------------------------------------------#
#  setup starts here                                                         #
# ---------------------------------------------------------------------------#

from setuptools import setup, find_packages

setup(

    name='bob',
    version=BOB['version'],
    description='Bob is a free signal-processing and machine learning toolbox',
    long_description=open('README.rst').read(),
    url='http://idiap.github.com/bob',
    download_url='http://www.idiap.ch/software/bob/packages/bob-%s.tar.gz' % BOB['version'],
    author='Bob Developers',
    author_email='bob-devel@googlegroups.com',
    keywords=['signal processing', 'machine learning', 'biometrics'],
    license='GPLv3',

    classifiers=[
      'Classifier: Development Status :: 5 - Production/Stable',
      'Classifier: Environment :: Console (Text Based)',
      'Classifier: Intended Audience :: Science/Research',
      'Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Classifier: Programming Language :: C++',
      'Classifier: Programming Language :: Python',
      'Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],

    packages=find_packages(),
    package_data={'bob': DATAFILES},
    include_package_data=True,
    zip_safe=False,

    ext_modules=EXTENSIONS,
    cmdclass = {'build_ext': build_ext},

    install_requires=[
      'setuptools',
      'numpy',
      'matplotlib',
      'sqlalchemy',
      'scipy',
      'nose',
      ] + EXTRA_REQUIREMENTS,

    entry_points={
      'console_scripts': CONSOLE_SCRIPTS,
      'bob.db': DATABASES,
      'nose.plugins.0.10': [
        'insulate = bob.core.test.insulate:Insulate',
        'insulateslave = bob.core.test.insulate:InsulateSlave',
        ]
      },

    )
