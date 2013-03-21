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

  =========== ===== ======= =======================================
   test        fmt   codec   figure (lower means better quality)           
  =========== ===== ======= =======================================
   color       mov   h264    4.603 min=0.890@22 max=8.387@9
   frameskip   mov   h264    0.108 min=0.009@11 max=0.344@0
   noise       mov   h264    44.900 min=43.916@4 max=46.103@29
   user        mov   h264    1.983 min=1.525@0 max=2.286@7 
  =========== ===== ======= =======================================

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

Example Output in Different Platforms
-------------------------------------

In what follows, you will find some tabbed output for different combinations of
operating systems and FFmpeg_/libav_ versions. To run these tests we only
executed:

.. code-block:: sh

  $ bob_video_test.py

Idiap Linux (Xubuntu), version 12.10 + libav 0.8.3
==================================================

You can find images and videos generated by the program `at this URL (Idiap 12.10 + libav 0.8.3) <http://www.idiap.ch/software/bob/docs/extras/video/idiap-12.10+libav-0.8.3/>`_.

=========== ===== ============ ================================================
 test        fmt     codec      figure (lower means better quality)           
=========== ===== ============ ================================================
 color       avi   ffv1         4.569 min=0.888\@22 max=8.377\@9                
 color       avi   h264         4.603 min=0.890\@22 max=8.388\@9                
 color       avi   libvpx       4.657 min=0.955\@26 max=8.528\@9                
 color       avi   libx264      4.603 min=0.890\@22 max=8.388\@9                
 color       avi   mjpeg        4.676 min=0.965\@22 max=8.469\@9                
 color       avi   mpeg1video   4.781 min=1.103\@28 max=8.483\@9                
 color       avi   mpeg2video   4.741 min=1.004\@16 max=8.466\@9                
 color       avi   mpeg4        4.892 min=1.087\@24 max=8.658\@9                
 color       avi   msmpeg4      4.921 min=1.073\@24 max=8.717\@9                
 color       avi   msmpeg4v2    4.921 min=1.073\@24 max=9.181\@17               
 color       avi   vp8          4.657 min=0.955\@26 max=8.528\@9                
 color       avi   wmv1         4.871 min=1.087\@24 max=8.729\@9                
 color       avi   wmv2         4.884 min=1.093\@24 max=8.823\@9                
 color       avi   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 color       mov   ffv1         4.569 min=0.888\@22 max=8.377\@9                
 color       mov   h264         4.603 min=0.890\@22 max=8.387\@9                
 color       mov   libvpx       4.657 min=0.955\@26 max=8.528\@9                
 color       mov   libx264      4.603 min=0.890\@22 max=8.387\@9                
 color       mov   mjpeg        4.676 min=0.965\@22 max=8.469\@9                
 color       mov   mpeg1video   4.781 min=1.103\@28 max=8.483\@9                
 color       mov   mpeg2video   4.741 min=1.004\@16 max=8.466\@9                
 color       mov   mpeg4        4.892 min=1.087\@24 max=8.658\@9                
 color       mov   msmpeg4      4.921 min=1.073\@24 max=8.717\@9                
 color       mov   msmpeg4v2    4.921 min=1.073\@24 max=9.181\@17               
 color       mov   vp8          4.657 min=0.955\@26 max=8.528\@9                
 color       mov   wmv1         4.871 min=1.087\@24 max=8.729\@9                
 color       mov   wmv2         4.884 min=1.093\@24 max=8.823\@9                
 color       mov   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 color       mp4   ffv1         format+codec unsupported                      
 color       mp4   h264         4.603 min=0.890\@22 max=8.387\@9                
 color       mp4   libvpx       format+codec unsupported                      
 color       mp4   libx264      4.603 min=0.890\@22 max=8.387\@9                
 color       mp4   mjpeg        4.676 min=0.965\@22 max=8.469\@9                
 color       mp4   mpeg1video   4.781 min=1.103\@28 max=8.483\@9                
 color       mp4   mpeg2video   4.741 min=1.004\@16 max=8.466\@9                
 color       mp4   mpeg4        4.892 min=1.087\@24 max=8.658\@9                
 color       mp4   msmpeg4      format+codec unsupported                      
 color       mp4   msmpeg4v2    format+codec unsupported                      
 color       mp4   vp8          format+codec unsupported                      
 color       mp4   wmv1         format+codec unsupported                      
 color       mp4   wmv2         format+codec unsupported                      
 color       mp4   zlib         format+codec unsupported                      
 frameskip   avi   ffv1         0.018 min=0.002\@11 max=0.029\@8                
 frameskip   avi   h264         0.108 min=0.009\@11 max=0.344\@0                
 frameskip   avi   libvpx       0.129 min=0.042\@11 max=0.198\@8                
 frameskip   avi   libx264      0.108 min=0.009\@11 max=0.344\@0                
 frameskip   avi   mjpeg        0.380 min=0.141\@11 max=1.108\@0                
 frameskip   avi   mpeg1video   0.426 min=0.237\@17 max=1.338\@0                
 frameskip   avi   mpeg2video   0.411 min=0.223\@15 max=1.284\@0                
 frameskip   avi   mpeg4        0.454 min=0.263\@17 max=0.858\@0                
 frameskip   avi   msmpeg4      1.684 min=0.257\@12 max=3.766\@15               
 frameskip   avi   msmpeg4v2    1.683 min=0.257\@12 max=3.765\@15               
 frameskip   avi   vp8          0.129 min=0.042\@11 max=0.198\@8                
 frameskip   avi   wmv1         0.627 min=0.191\@11 max=1.568\@8                
 frameskip   avi   wmv2         0.626 min=0.191\@11 max=1.566\@8                
 frameskip   avi   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 frameskip   mov   ffv1         0.018 min=0.002\@11 max=0.029\@8                
 frameskip   mov   h264         0.108 min=0.009\@11 max=0.344\@0                
 frameskip   mov   libvpx       0.129 min=0.042\@11 max=0.198\@8                
 frameskip   mov   libx264      0.108 min=0.009\@11 max=0.344\@0                
 frameskip   mov   mjpeg        0.380 min=0.141\@11 max=1.108\@0                
 frameskip   mov   mpeg1video   0.426 min=0.237\@17 max=1.338\@0                
 frameskip   mov   mpeg2video   0.411 min=0.223\@15 max=1.284\@0                
 frameskip   mov   mpeg4        0.454 min=0.263\@17 max=0.858\@0                
 frameskip   mov   msmpeg4      1.684 min=0.257\@12 max=3.766\@15               
 frameskip   mov   msmpeg4v2    1.683 min=0.257\@12 max=3.765\@15               
 frameskip   mov   vp8          0.129 min=0.042\@11 max=0.198\@8                
 frameskip   mov   wmv1         0.627 min=0.191\@11 max=1.568\@8                
 frameskip   mov   wmv2         0.626 min=0.191\@11 max=1.566\@8                
 frameskip   mov   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 frameskip   mp4   ffv1         format+codec unsupported                      
 frameskip   mp4   h264         0.108 min=0.009\@11 max=0.344\@0                
 frameskip   mp4   libvpx       format+codec unsupported                      
 frameskip   mp4   libx264      0.108 min=0.009\@11 max=0.344\@0                
 frameskip   mp4   mjpeg        0.380 min=0.141\@11 max=1.108\@0                
 frameskip   mp4   mpeg1video   0.426 min=0.237\@17 max=1.338\@0                
 frameskip   mp4   mpeg2video   0.411 min=0.223\@15 max=1.284\@0                
 frameskip   mp4   mpeg4        0.454 min=0.263\@17 max=0.858\@0                
 frameskip   mp4   msmpeg4      format+codec unsupported                      
 frameskip   mp4   msmpeg4v2    format+codec unsupported                      
 frameskip   mp4   vp8          format+codec unsupported                      
 frameskip   mp4   wmv1         format+codec unsupported                      
 frameskip   mp4   wmv2         format+codec unsupported                      
 frameskip   mp4   zlib         format+codec unsupported                      
 noise       avi   ffv1         44.192 min=43.887\@0 max=44.568\@8              
 noise       avi   h264         44.882 min=43.738\@2 max=45.848\@27             
 noise       avi   libvpx       48.629 min=44.156\@12 max=54.365\@27            
 noise       avi   libx264      44.883 min=44.089\@2 max=45.857\@29             
 noise       avi   mjpeg        45.723 min=43.942\@3 max=48.283\@28             
 noise       avi   mpeg1video   46.270 min=44.412\@2 max=48.632\@29             
 noise       avi   mpeg2video   45.227 min=44.008\@5 max=48.528\@29             
 noise       avi   mpeg4        45.769 min=43.720\@4 max=48.472\@27             
 noise       avi   msmpeg4      45.757 min=44.034\@7 max=48.055\@24             
 noise       avi   msmpeg4v2    45.789 min=43.908\@6 max=48.423\@27             
 noise       avi   vp8          48.796 min=43.765\@0 max=50.864\@15             
 noise       avi   wmv1         45.729 min=43.878\@6 max=47.921\@29             
 noise       avi   wmv2         46.105 min=44.205\@3 max=48.261\@28             
 noise       avi   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 noise       mov   ffv1         44.200 min=43.869\@20 max=44.719\@22            
 noise       mov   h264         44.882 min=43.991\@6 max=46.183\@27             
 noise       mov   libvpx       48.692 min=43.934\@0 max=50.906\@15             
 noise       mov   libx264      44.909 min=43.773\@3 max=46.079\@29             
 noise       mov   mjpeg        45.754 min=43.823\@8 max=48.278\@28             
 noise       mov   mpeg1video   46.353 min=44.326\@1 max=48.712\@29             
 noise       mov   mpeg2video   45.970 min=43.952\@4 max=50.645\@29             
 noise       mov   mpeg4        45.772 min=43.961\@4 max=48.414\@28             
 noise       mov   msmpeg4      45.764 min=43.867\@5 max=48.156\@29             
 noise       mov   msmpeg4v2    45.844 min=44.009\@6 max=48.317\@27             
 noise       mov   vp8          48.323 min=43.985\@12 max=50.512\@19            
 noise       mov   wmv1         45.803 min=44.109\@3 max=48.334\@29             
 noise       mov   wmv2         46.081 min=43.950\@4 max=48.293\@26             
 noise       mov   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 noise       mp4   ffv1         format+codec unsupported                      
 noise       mp4   h264         44.856 min=43.749\@1 max=46.045\@27             
 noise       mp4   libvpx       format+codec unsupported                      
 noise       mp4   libx264      44.785 min=43.820\@0 max=46.093\@28             
 noise       mp4   mjpeg        45.725 min=43.979\@7 max=48.208\@28             
 noise       mp4   mpeg1video   46.227 min=44.144\@2 max=48.241\@27             
 noise       mp4   mpeg2video   46.060 min=43.991\@5 max=51.358\@29             
 noise       mp4   mpeg4        45.690 min=44.072\@6 max=47.974\@28             
 noise       mp4   msmpeg4      format+codec unsupported                      
 noise       mp4   msmpeg4v2    format+codec unsupported                      
 noise       mp4   vp8          format+codec unsupported                      
 noise       mp4   wmv1         format+codec unsupported                      
 noise       mp4   wmv2         format+codec unsupported                      
 noise       mp4   zlib         format+codec unsupported                      
 user        avi   ffv1         1.174 min=1.166\@2 max=1.187\@7                 
 user        avi   h264         1.988 min=1.525\@0 max=2.290\@7                 
 user        avi   libvpx       1.614 min=1.464\@0 max=1.711\@8                 
 user        avi   libx264      1.988 min=1.525\@0 max=2.290\@7                 
 user        avi   mjpeg        1.067 min=1.014\@2 max=1.444\@0                 
 user        avi   mpeg1video   1.586 min=1.447\@1 max=1.895\@0                 
 user        avi   mpeg2video   1.743 min=1.515\@3 max=1.891\@8                 
 user        avi   mpeg4        1.794 min=1.606\@1 max=1.906\@9                 
 user        avi   msmpeg4      1.802 min=1.599\@1 max=1.925\@8                 
 user        avi   msmpeg4v2    1.775 min=1.599\@1 max=1.868\@9                 
 user        avi   vp8          1.614 min=1.464\@0 max=1.711\@8                 
 user        avi   wmv1         1.802 min=1.599\@1 max=1.925\@8                 
 user        avi   wmv2         1.799 min=1.596\@1 max=1.921\@8                 
 user        avi   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 user        mov   ffv1         1.174 min=1.166\@2 max=1.187\@7                 
 user        mov   h264         1.983 min=1.525\@0 max=2.286\@7                 
 user        mov   libvpx       1.614 min=1.464\@0 max=1.711\@8                 
 user        mov   libx264      1.983 min=1.525\@0 max=2.286\@7                 
 user        mov   mjpeg        1.067 min=1.014\@2 max=1.444\@0                 
 user        mov   mpeg1video   1.586 min=1.447\@1 max=1.895\@0                 
 user        mov   mpeg2video   1.743 min=1.515\@3 max=1.891\@8                 
 user        mov   mpeg4        1.794 min=1.606\@1 max=1.906\@9                 
 user        mov   msmpeg4      1.802 min=1.599\@1 max=1.925\@8                 
 user        mov   msmpeg4v2    1.775 min=1.599\@1 max=1.868\@9                 
 user        mov   vp8          1.614 min=1.464\@0 max=1.711\@8                 
 user        mov   wmv1         1.802 min=1.599\@1 max=1.925\@8                 
 user        mov   wmv2         1.799 min=1.596\@1 max=1.921\@8                 
 user        mov   zlib         0.000 min=0.000\@0 max=0.000\@0                 
 user        mp4   ffv1         format+codec unsupported                      
 user        mp4   h264         1.983 min=1.525\@0 max=2.286\@7                 
 user        mp4   libvpx       format+codec unsupported                      
 user        mp4   libx264      1.983 min=1.525\@0 max=2.286\@7                 
 user        mp4   mjpeg        1.067 min=1.014\@2 max=1.444\@0                 
 user        mp4   mpeg1video   1.586 min=1.447\@1 max=1.895\@0                 
 user        mp4   mpeg2video   1.743 min=1.515\@3 max=1.891\@8                 
 user        mp4   mpeg4        1.794 min=1.606\@1 max=1.906\@9                 
 user        mp4   msmpeg4      format+codec unsupported                      
 user        mp4   msmpeg4v2    format+codec unsupported                      
 user        mp4   vp8          format+codec unsupported                      
 user        mp4   wmv1         format+codec unsupported                      
 user        mp4   wmv2         format+codec unsupported                      
 user        mp4   zlib         format+codec unsupported                      
=========== ===== ============ ================================================

MacOSX 10.8.3 + FFmpeg 1.1.2
============================

You can find images and videos generated by the program `at this URL (MacOSX 10.8.3 + FFmpeg 1.1.2) <http://www.idiap.ch/software/bob/docs/extras/video/osx-10.8.3+ffmpeg-1.1.2/>`_.

=========== ===== ================== ========================================
 test        fmt   codec              figure (lower is better quality)
=========== ===== ================== ========================================
 color       avi   ffv1               4.643 min=0.999\@24 max=8.420\@9
 color       avi   h264               4.685 min=1.001\@24 max=8.473\@9
 color       avi   libvpx             4.736 min=1.079\@26 max=8.503\@9
 color       avi   libx264            4.685 min=1.001\@24 max=8.473\@9
 color       avi   mjpeg              4.617 min=0.934\@24 max=8.440\@9
 color       avi   mpeg1video         4.820 min=1.125\@16 max=8.548\@9
 color       avi   mpeg2video         4.787 min=1.130\@16 max=8.465\@9
 color       avi   mpeg4              4.956 min=1.129\@24 max=8.725\@9
 color       avi   mpegvideo          4.787 min=1.130\@16 max=8.465\@9
 color       avi   msmpeg4            4.987 min=1.114\@24 max=8.731\@9
 color       avi   msmpeg4v2          4.949 min=1.114\@24 max=8.667\@9
 color       avi   vp8                4.736 min=1.079\@26 max=8.503\@9
 color       avi   wmv1               4.925 min=1.129\@24 max=8.728\@9
 color       avi   wmv2               4.936 min=1.138\@24 max=8.796\@9
 color       avi   zlib               0.000 min=0.000\@0 max=0.000\@0
 color       mov   ffv1               4.643 min=0.999\@24 max=8.420\@9
 color       mov   h264               4.645 min=1.001\@24 max=8.424\@9
 color       mov   libvpx             4.736 min=1.079\@26 max=8.503\@9
 color       mov   libx264            4.645 min=1.001\@24 max=8.424\@9
 color       mov   mjpeg              4.617 min=0.934\@24 max=8.440\@9
 color       mov   mpeg1video         4.820 min=1.125\@16 max=8.548\@9
 color       mov   mpeg2video         4.787 min=1.130\@16 max=8.465\@9
 color       mov   mpeg4              4.956 min=1.129\@24 max=8.725\@9
 color       mov   mpegvideo          4.787 min=1.130\@16 max=8.465\@9
 color       mov   msmpeg4            4.987 min=1.114\@24 max=8.731\@9
 color       mov   msmpeg4v2          4.949 min=1.114\@24 max=8.667\@9
 color       mov   vp8                4.736 min=1.079\@26 max=8.503\@9
 color       mov   wmv1               4.925 min=1.129\@24 max=8.728\@9
 color       mov   wmv2               4.936 min=1.138\@24 max=8.796\@9
 color       mov   zlib               0.000 min=0.000\@0 max=0.000\@0
 color       mp4   ffv1               format+codec unsupported
 color       mp4   h264               4.645 min=1.001\@24 max=8.424\@9
 color       mp4   libvpx             format+codec unsupported
 color       mp4   libx264            4.645 min=1.001\@24 max=8.424\@9
 color       mp4   mjpeg              4.617 min=0.934\@24 max=8.440\@9
 color       mp4   mpeg1video         4.820 min=1.125\@16 max=8.548\@9
 color       mp4   mpeg2video         4.787 min=1.130\@16 max=8.465\@9
 color       mp4   mpeg4              4.956 min=1.129\@24 max=8.725\@9
 color       mp4   mpegvideo          4.787 min=1.130\@16 max=8.465\@9
 color       mp4   msmpeg4            format+codec unsupported
 color       mp4   msmpeg4v2          format+codec unsupported
 color       mp4   vp8                format+codec unsupported
 color       mp4   wmv1               format+codec unsupported
 color       mp4   wmv2               format+codec unsupported
 color       mp4   zlib               format+codec unsupported
 frameskip   avi   ffv1               0.018 min=0.002\@11 max=0.029\@8
 frameskip   avi   h264               0.120 min=0.017\@21 max=0.300\@0
 frameskip   avi   libvpx             0.122 min=0.051\@11 max=0.181\@0
 frameskip   avi   libx264            0.120 min=0.017\@21 max=0.300\@0
 frameskip   avi   mjpeg              0.386 min=0.147\@11 max=1.085\@0
 frameskip   avi   mpeg1video         0.427 min=0.243\@11 max=1.310\@0
 frameskip   avi   mpeg2video         0.408 min=0.229\@17 max=1.258\@0
 frameskip   avi   mpeg4              0.456 min=0.253\@12 max=0.849\@0
 frameskip   avi   mpegvideo          0.408 min=0.229\@17 max=1.258\@0
 frameskip   avi   msmpeg4            1.608 min=0.434\@24 max=3.409\@25
 frameskip   avi   msmpeg4v2          1.600 min=0.434\@24 max=3.708\@5
 frameskip   avi   vp8                0.122 min=0.051\@11 max=0.181\@0
 frameskip   avi   wmv1               0.617 min=0.191\@11 max=1.981\@8
 frameskip   avi   wmv2               0.614 min=0.191\@11 max=1.978\@8
 frameskip   avi   zlib               0.000 min=0.000\@0 max=0.000\@0
 frameskip   mov   ffv1               0.018 min=0.002\@11 max=0.029\@8
 frameskip   mov   h264               0.042 min=0.011\@11 max=0.085\@9
 frameskip   mov   libvpx             0.122 min=0.051\@11 max=0.181\@0
 frameskip   mov   libx264            0.042 min=0.011\@11 max=0.085\@9
 frameskip   mov   mjpeg              0.386 min=0.147\@11 max=1.085\@0
 frameskip   mov   mpeg1video         0.427 min=0.243\@11 max=1.310\@0
 frameskip   mov   mpeg2video         0.408 min=0.229\@17 max=1.258\@0
 frameskip   mov   mpeg4              0.456 min=0.253\@12 max=0.849\@0
 frameskip   mov   mpegvideo          0.408 min=0.229\@17 max=1.258\@0
 frameskip   mov   msmpeg4            1.608 min=0.434\@24 max=3.409\@25
 frameskip   mov   msmpeg4v2          1.600 min=0.434\@24 max=3.708\@5
 frameskip   mov   vp8                0.122 min=0.051\@11 max=0.181\@0
 frameskip   mov   wmv1               0.617 min=0.191\@11 max=1.981\@8
 frameskip   mov   wmv2               0.614 min=0.191\@11 max=1.978\@8
 frameskip   mov   zlib               0.000 min=0.000\@0 max=0.000\@0
 frameskip   mp4   ffv1               format+codec unsupported
 frameskip   mp4   h264               0.042 min=0.011\@11 max=0.085\@9
 frameskip   mp4   libvpx             format+codec unsupported
 frameskip   mp4   libx264            0.042 min=0.011\@11 max=0.085\@9
 frameskip   mp4   mjpeg              0.386 min=0.147\@11 max=1.085\@0
 frameskip   mp4   mpeg1video         0.427 min=0.243\@11 max=1.310\@0
 frameskip   mp4   mpeg2video         0.408 min=0.229\@17 max=1.258\@0
 frameskip   mp4   mpeg4              0.456 min=0.253\@12 max=0.849\@0
 frameskip   mp4   mpegvideo          0.408 min=0.229\@17 max=1.258\@0
 frameskip   mp4   msmpeg4            format+codec unsupported
 frameskip   mp4   msmpeg4v2          format+codec unsupported
 frameskip   mp4   vp8                format+codec unsupported
 frameskip   mp4   wmv1               format+codec unsupported
 frameskip   mp4   wmv2               format+codec unsupported
 frameskip   mp4   zlib               format+codec unsupported
 noise       avi   ffv1               44.108 min=43.717\@16 max=44.565\@22
 noise       avi   h264               44.509 min=43.859\@4 max=45.146\@27
 noise       avi   libvpx             46.882 min=43.812\@1 max=49.422\@18
 noise       avi   libx264            44.572 min=43.917\@5 max=45.236\@29
 noise       avi   mjpeg              45.739 min=43.819\@4 max=48.211\@29
 noise       avi   mpeg1video         46.320 min=44.273\@3 max=48.996\@29
 noise       avi   mpeg2video         46.054 min=43.987\@7 max=51.580\@29
 noise       avi   mpeg4              45.755 min=44.071\@2 max=48.502\@28
 noise       avi   mpegvideo          44.951 min=43.775\@1 max=46.796\@24
 noise       avi   msmpeg4            45.749 min=43.934\@5 max=48.267\@29
 noise       avi   msmpeg4v2          45.846 min=43.987\@0 max=48.264\@27
 noise       avi   vp8                46.457 min=43.931\@12 max=48.857\@27
 noise       avi   wmv1               45.804 min=44.219\@10 max=48.252\@28
 noise       avi   wmv2               46.091 min=44.113\@3 max=48.380\@25
 noise       avi   zlib               0.000 min=0.000\@0 max=0.000\@0
 noise       mov   ffv1               44.128 min=43.657\@15 max=44.513\@21
 noise       mov   h264               44.168 min=43.794\@24 max=44.577\@7
 noise       mov   libvpx             47.009 min=44.127\@4 max=49.547\@17
 noise       mov   libx264            44.143 min=43.813\@23 max=44.529\@16
 noise       mov   mjpeg              44.378 min=44.020\@18 max=44.670\@0
 noise       mov   mpeg1video         44.564 min=43.903\@9 max=45.314\@0
 noise       mov   mpeg2video         44.340 min=44.021\@26 max=44.733\@0
 noise       mov   mpeg4              44.338 min=43.923\@3 max=44.677\@11
 noise       mov   mpegvideo          44.343 min=43.978\@8 max=44.904\@29
 noise       mov   msmpeg4            44.293 min=43.870\@9 max=44.669\@24
 noise       mov   msmpeg4v2          44.256 min=43.859\@5 max=44.596\@21
 noise       mov   vp8                47.558 min=43.955\@0 max=52.720\@25
 noise       mov   wmv1               44.283 min=43.848\@24 max=44.643\@14
 noise       mov   wmv2               44.323 min=43.957\@10 max=44.727\@0
 noise       mov   zlib               0.000 min=0.000\@0 max=0.000\@0
 noise       mp4   ffv1               format+codec unsupported
 noise       mp4   h264               44.118 min=43.717\@18 max=44.439\@1
 noise       mp4   libvpx             format+codec unsupported
 noise       mp4   libx264            44.218 min=43.870\@8 max=44.730\@19
 noise       mp4   mjpeg              44.374 min=44.061\@2 max=44.902\@0
 noise       mp4   mpeg1video         44.537 min=44.157\@18 max=45.222\@0
 noise       mp4   mpeg2video         44.397 min=43.834\@5 max=44.825\@0
 noise       mp4   mpeg4              44.276 min=43.875\@9 max=44.912\@17
 noise       mp4   mpegvideo          44.339 min=43.812\@2 max=45.328\@0
 noise       mp4   msmpeg4            format+codec unsupported
 noise       mp4   msmpeg4v2          format+codec unsupported
 noise       mp4   vp8                format+codec unsupported
 noise       mp4   wmv1               format+codec unsupported
 noise       mp4   wmv2               format+codec unsupported
 noise       mp4   zlib               format+codec unsupported
 user        avi   ffv1               1.463 min=1.457\@5 max=1.472\@7
 user        avi   h264               2.028 min=1.666\@0 max=2.201\@9
 user        avi   libvpx             1.999 min=1.646\@0 max=2.420\@2
 user        avi   libx264            2.028 min=1.666\@0 max=2.201\@9
 user        avi   mjpeg              1.197 min=1.149\@6 max=1.532\@0
 user        avi   mpeg1video         1.760 min=1.641\@1 max=2.061\@0
 user        avi   mpeg2video         1.882 min=1.694\@3 max=2.026\@0
 user        avi   mpeg4              1.960 min=1.782\@1 max=2.076\@9
 user        avi   mpegvideo          1.882 min=1.694\@3 max=2.026\@0
 user        avi   msmpeg4            1.964 min=1.773\@1 max=2.088\@8
 user        avi   msmpeg4v2          1.921 min=1.773\@1 max=2.008\@9
 user        avi   vp8                1.999 min=1.646\@0 max=2.420\@2
 user        avi   wmv1               1.964 min=1.773\@1 max=2.088\@8
 user        avi   wmv2               1.958 min=1.768\@1 max=2.082\@8
 user        avi   zlib               0.000 min=0.000\@0 max=0.000\@0
 user        mov   ffv1               1.463 min=1.457\@5 max=1.472\@7
 user        mov   h264               1.533 min=1.477\@0 max=1.566\@7
 user        mov   libvpx             2.103 min=1.646\@0 max=2.547\@2
 user        mov   libx264            1.533 min=1.477\@0 max=1.566\@7
 user        mov   mjpeg              1.197 min=1.149\@6 max=1.532\@0
 user        mov   mpeg1video         1.760 min=1.641\@1 max=2.061\@0
 user        mov   mpeg2video         1.882 min=1.694\@3 max=2.026\@0
 user        mov   mpeg4              1.960 min=1.782\@1 max=2.076\@9
 user        mov   mpegvideo          1.882 min=1.694\@3 max=2.026\@0
 user        mov   msmpeg4            1.964 min=1.773\@1 max=2.088\@8
 user        mov   msmpeg4v2          1.921 min=1.773\@1 max=2.008\@9
 user        mov   vp8                2.103 min=1.646\@0 max=2.547\@2
 user        mov   wmv1               1.964 min=1.773\@1 max=2.088\@8
 user        mov   wmv2               1.958 min=1.768\@1 max=2.082\@8
 user        mov   zlib               0.000 min=0.000\@0 max=0.000\@0
 user        mp4   ffv1               format+codec unsupported
 user        mp4   h264               1.533 min=1.477\@0 max=1.566\@7
 user        mp4   libvpx             format+codec unsupported
 user        mp4   libx264            1.533 min=1.477\@0 max=1.566\@7
 user        mp4   mjpeg              1.197 min=1.149\@6 max=1.532\@0
 user        mp4   mpeg1video         1.760 min=1.641\@1 max=2.061\@0
 user        mp4   mpeg2video         1.882 min=1.694\@3 max=2.026\@0
 user        mp4   mpeg4              1.960 min=1.782\@1 max=2.076\@9
 user        mp4   mpegvideo          1.882 min=1.694\@3 max=2.026\@0
 user        mp4   msmpeg4            format+codec unsupported
 user        mp4   msmpeg4v2          format+codec unsupported
 user        mp4   vp8                format+codec unsupported
 user        mp4   wmv1               format+codec unsupported
 user        mp4   wmv2               format+codec unsupported
 user        mp4   zlib               format+codec unsupported
=========== ===== ================== ========================================

.. include:: links.rst

.. Place here your external references
