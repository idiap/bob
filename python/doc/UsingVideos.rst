.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed 20 Mar 2013 11:30:02 CET 
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

=============================
 Using Videos with |project|
=============================

Video read and write support in |project| uses FFmpeg_ as implementation
backend. In Ubuntu-based distributions, FFmpeg_ was replaced by libav_, which
is a fork based on FFmpeg_ version 0.8. |project| can detect and use libav_
when FFmpeg_ is not available on the machine. We currently support a variety of
FFmpeg_ (and libav_) releases, ranging from FFmpeg_ 0.5 until the most recent
branches.

FFmpeg_ (and libav_) provide a (reasonably) uniform API for reading and writing
data into a variety of video container formats, using different video and audio
codecs. |project| leverages on this API to propose a sub-range of formats and
codecs that work well together, with low distortion patterns and accross
platforms.

.. note::

  As much as we strive to make video formats and codecs available to all
  platforms in which |project| is available, codecs, in particular may be
  disabled by compilation options on FFmpeg_ or libav_, in which case |project|
  builds will not be able to use them.

.. note::

  Currently, |project| does not support reading or writing of audio streams on
  video data - only images.

This section provides guidance in choosing a set of formats and codecs for your
project, so you will be able to leverage the maximum from |project|.

Codec and Format Availability
-----------------------------

To get a list of all FFmpeg_ (or libav_) supported formats for a given build of
|project|, use the ``bob_video_test.py`` application:

.. code-block:: sh

  $ bob_video_test.py --list-all-codecs # lists all codecs available
  
  $ bob_video_test.py --list-all-formats # lists all formats available

These listings represent all that is compiled with your current installation of
FFmpeg_ or libav_. To list supported formats and codecs by |project| use
another set of command-line options:

.. code-block:: sh

  $ bob_video_test.py --list-codecs # lists all codecs currently supported
  
  $ bob_video_test.py --list-formats # lists all formats currently supported

The program ``bob_video_test.py`` can be used to run a sequence of tests using
all combinations of *supported* formats and tests:

.. code-block:: sh

  $ bob_video_test.py # runs all tests

This will run through all combinations of supported codecs and formats and will
report average distortion figures for each of 4 different tests, which exercise
different aspects of each combination of format and codec. Here is a an example
output:

.. code-block:: text

  Video Encoding/Decoding Test Tool v1.2.0a0 (bob_video_test)
  Settings:
    Width    : 128 pixels
    Height   : 128 pixels
    Length   : 30 frames
    Framerate: 30.000000 Hz
  Legend:
    C: Color test
    N: Noise test
    U: User test
    S: Frameskip test
  Running 4 test(s)...CSNU

   test      | fmt | codec            | figure (lower is better quality)
  -----------+-----+------------------+-----------------------------------
   color     | mov | h264             | 4.603 min=0.890@22 max=8.387@9
   frameskip | mov | h264             | 0.108 min=0.009@11 max=0.344@0
   noise     | mov | h264             | 44.900 min=43.916@4 max=46.103@29
   user      | mov | h264             | 1.983 min=1.525@0 max=2.286@7 

Each line in the output table represents the average distortion patterns for
the particular test using the format and codec described. The lower the
distortion, the better the combination of format and codecs is. Different tests
have different levels of baseline performance. The figures above were obtained
in a healthy (no know bugs) system, running libav_ 0.8.13 (Ubuntu 12.10). Each
line indicates, besides the average distortion per frame, the minimum and the
maximum obtained and in which frame (counting from 0 - zero), that figure was
obtained.

The video tests are made on temporary files that are discarded after the test
is completed. You can use the option ``--output=<directory>`` to specify a
directory in which the generated files will be saved. You can then go to these
directories and explore potential problems you may find.

Each test creates a video from an artificially generated test signal, encodes
it using the defined format and codec and reads it back, comparing the output
result with the original sequence. The sole exception is the ``user`` test. In
this test, a user test sequence is (partially) loaded and tested. If you don't
specify any sequence, a default sequence from |project| is used. If you want to
test a specific sequence of your own, use ``--user-video`` to specify the path
of the video sequence you want to test with. By default, only the first 10
frames of the sequence are used for the test, to speed-up execution. You can
change this behavior with the option ``--user-frames``. Here is an example:

.. code-block:: sh

  $ bob_video_test.py --user-video=test_sample.avi

All tests are executed by default, on all combination of formats and codecs.
That can be long. You can limit the test execution by properly choosing the
format (``--format``), the codec (``--codec``) and the tests to execute. For
example:

.. code-block:: sh
  
  # execute only the user video test with a user provided video and
  # using the H.264 built-in codec and a MOV output file format.
  $ bob_video_test.py --format mov --codec h264 --user-video=test_sample.avi -- user

.. note::

  Not all codecs can be used by all formats available. For example, the ``mp4``
  file format cannot use the ``vp8`` codec, even if both are supported by
  |project|. To know which formats support each codec, you can execute the
  following python code:

  .. code-block:: python

    import bob
    bob.io.supported_videowriter_formats()['mp4']['supported_codecs'].keys()
    ['h264', 'libx264', 'mjpeg', 'mpeg1video']

  You can see from the output command that only 4 codecs are supported by the
  file format ``mp4``.

You can test new combinations of formats and codecs which are not currently
supported by |project|, as long as they are supported by the underlying FFmpeg_
or libav_ installations. In this case, just specify the format and/or codec
names using ``--format`` and ``--codec`` options in the application
``bob_video_test.py``. The advantage of using *supported* formats and codecs is
that we make sure a minimal distortion figure is respected in all platform
nightly builds, with our unit and integration tests. We cannot, currently,
test all possible combinations of codecs and formats.

Know Your Platforms
-------------------

One important aspect when working with videos is to know there will be some
lossy compression applied to the output. This means you will **loose**
information when re-encoding. When working with videos, you will want to choose
the combination of format and codec that will work well accross different
platforms. We recommend to run ``bob_video_test.py`` with a few of your video
inputs to make sure they can be decoded with low distortion where you plan to
work.

.. note::

  The only codec that supports lossless compression in |project| is ``zlib``.
  Of course, the output files are considerably bigger, but they continue to be
  readable using any FFmpeg_-based movie player or even QuickTime (on OSX), if
  Perian is installed.

.. include:: links.rst

.. Place here your external references
