#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 14 Mar 17:53:16 2013 

"""This program can run manual tests using any video codec available in Bob. It
can report standard distortion figures and build video test sequences for
manual inspection. It tries to help identifying problems with:

  1. Color distortion
  2. Frame skipping or delay
  3. Encoding or decoding quality
  4. User test (with a user provided video sample)

You can parameterize the program with the type of file, (FFmpeg) codec and a
few other parameters. The program then generates artificial input signals to
test for each of the parameters above.
"""

import os
import sys
import argparse
import numpy

# internal
from .. import supported_video_codecs, available_video_codecs, supported_videowriter_formats, available_videowriter_formats
from .. import utils, create_directories_save
from ... import version
from .. import save as save_to_file
from ...test import utils as test_utils
from ...io import test as io_test

CODECS = supported_video_codecs()
ALL_CODECS = available_video_codecs()

def list_codecs(*args, **kwargs):
  retval = """\
  Supported Codecs:
  -----------------\n"""

  for k in sorted(CODECS.keys()):
    retval += ("  %-20s  %s\n" % (k, CODECS[k]['long_name']))[:80]

  return retval[:-1]

def list_all_codecs(*args, **kwargs):
  retval = """\
  Available Codecs:
  -----------------\n"""

  for k in sorted(ALL_CODECS.keys()):
    retval += ("  %-20s  %s\n" % (k, ALL_CODECS[k]['long_name']))[:80]

  return retval[:-1]

FORMATS = supported_videowriter_formats()
ALL_FORMATS = available_videowriter_formats()

def list_formats(*args, **kwargs):

  retval = """\
  Supported Formats:
  ------------------\n"""

  for k in sorted(FORMATS.keys()):
    retval += ("  %-20s  %s\n" % (k, FORMATS[k]['long_name']))[:80]

  return retval[:-1]

def list_all_formats(*args, **kwargs):

  retval = """\
  Available Formats:
  ------------------\n"""

  for k in sorted(ALL_FORMATS.keys()):
    retval += ("  %-20s  %s\n" % (k, ALL_FORMATS[k]['long_name']))[:80]

  return retval[:-1]

__epilog__ = """Example usage:

1. Check for color distortion using H.264 codec in a .mov video container:

  $ %(prog)s --format='mov' --codec='h264' color

2. Check for frame skipping using MJPEG codec in an .avi video container:

  $ %(prog)s --format='avi' --codec='mjpeg' frameskip

3. Check for encoding/decoding quality using a FFV1 codec in a '.wmv' video
container:

  $ %(prog)s --format='wmv' --codec='ffv1' noise

4. To run-only the user-video test and provide a test video:

  $ %(prog)s --format='mov' --user-video=test_sample.avi user

5. To list all available codecs:

  $ %(prog)s --list-codecs

6. To list all available formats:

  $ %(prog)s --list-formats

7. Run all tests for all codecs and formats:

  $ %(prog)s
""" % {
    'prog': os.path.basename(sys.argv[0]),
    }

def user_test(original, max_frames, format, codec, filename):
  """Returns distortion patterns for a set of frames with moving colors.

  Keyword parameters:

  original
    The name (path) to the original user file that will be used for the test

  max_frames
    The maximum number of frames to read from user input

  format
    The string that identifies the format to be used for the output file

  codec
    The codec to be used for the output file

  filename
    The name (path) of the file to use for encoding the test
  """

  from .. import VideoReader, VideoWriter
  vreader = VideoReader(original, check=True)
  orig = vreader[:max_frames]
  
  # rounding frame rate - some older codecs do not accept random frame rates
  framerate = vreader.frame_rate
  if codec in ('mpegvideo', 'mpeg1video', 'mpeg2video'):
    import math
    framerate = math.ceil(vreader.frame_rate)

  vwriter = VideoWriter(filename, vreader.height, vreader.width,
      framerate, codec=codec, format=format, check=False)
  for k in orig: vwriter.append(k)
  del vwriter
  return orig, framerate, VideoReader(filename, check=False)

def summarize(function, shape, framerate, format, codec, output=None):
  """Summarizes distortion patterns for a given set of video settings and 
  for a given input function.

  Keyword parameters:

  shape (int, int, int)
    The length (number of frames), height and width for the generated sequence

  format
    The string that identifies the format to be used for the output file

  codec
    The codec to be used for the output file

  output
    If set, the video is not created on the temporary directory, but it is 
    saved on the advised location. This must be a filename.

  Returns a single a single string summarizing the distortion results 
  """
  
  length, height, width = shape

  if output:
    fname = output
  else:
    fname = test_utils.temporary_filename(suffix='.%s' % format)

  retval = "did not run"

  try:
    # Width and height should be powers of 2 as the encoded image is going 
    # to be approximated to the closest one, would not not be the case. 
    # In this case, the encoding is subject to more noise as the filtered,
    # final image that is encoded will contain added noise on the extra
    # borders.
    orig, framerate, encoded = function(shape, framerate, format, codec, fname)

    tmp = []
    for k, of in enumerate(orig):
      tmp.append(abs(of.astype('float64')-encoded[k].astype('float64')).sum())
    size = numpy.prod(orig[0].shape)
    S = sum(tmp)/size
    M = S/len(tmp)
    Min = min(tmp)/size
    Max = max(tmp)/size
    ArgMin = tmp.index(min(tmp))
    ArgMax = tmp.index(max(tmp))
    retval = "%.3f min=%.3f@%d max=%.3f@%d" % (M, Min, ArgMin, Max, ArgMax)
    if abs(encoded.frame_rate - framerate) > 0.01:
      retval += " !FR(%g)" % abs(encoded.frame_rate - framerate)
    if len(encoded) != len(orig):
      retval += " !LEN(%d)" % len(encoded)

  finally:

    if os.path.exists(fname) and output is None: os.unlink(fname)

  if output:
    return retval, orig, encoded
  else:
    return retval

def detail(function, shape, framerate, format, codec, outdir):
  """Summarizes distortion patterns for a given set of video settings and 
  for a given input function.

  Keyword parameters:

  shape (int, int, int)
    The length (number of frames), height and width for the generated sequence

  format
    The string that identifies the format to be used for the output file

  codec
    The codec to be used for the output file

  outdir
    We will save all analysis for this sequence on the given output directory.

  Returns a single a single string summarizing the distortion results. 
  """

  length, height, width = shape
  
  text_format = "%%0%dd" % len(str(length-1))

  output = os.path.join(outdir, "video." + format)
  retval, orig, encoded = summarize(function, shape, framerate, 
      format, codec, output)
  
  length, _, height, width = orig.shape

  # save original, reloaded and difference images on output directories
  for i, orig_frame in enumerate(orig):
    out = numpy.ndarray((3, height, 3*width), dtype='uint8')
    out[:,:,:width] = orig_frame
    out[:,:,width:(2*width)] = encoded[i]
    diff = abs(encoded[i].astype('int64')-orig_frame.astype('int64'))
    diff[diff>0] = 255 #binary output
    out[:,:,(2*width):] = diff.astype('uint8')
    save_to_file(out, os.path.join(outdir, 'frame-' + (text_format%i) + '.png'))

  return retval

def main(user_input=None):

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  name = os.path.basename(os.path.splitext(sys.argv[0])[0])
  version_info = 'Video Encoding/Decoding Test Tool v%s (%s)' % (version, name)
  parser.add_argument('-V', '--version', action='version', version=version_info)

  test_choices = [
      'color',
      'frameskip',
      'noise',
      'user',
      ]

  parser.add_argument("test", metavar='TEST', type=str, nargs='*', 
      default=test_choices, help="The name of the test or tests you want to run. Choose between `%s'. If none given, run through all." % ('|'.join(test_choices)))
 
  supported_codecs = sorted(CODECS.keys())
  available_codecs = sorted(ALL_CODECS.keys())

  parser.add_argument("-c", "--codec", metavar='CODEC', type=str, nargs='*',
      default=supported_codecs, choices=available_codecs, help="The name of the codec you want to test with. For a list of available codecs, look below. If none given, run through all.")
  parser.add_argument("--list-codecs", action="store_true", default=False,
      help="List all supported codecs and exits")
  parser.add_argument("--list-all-codecs", action="store_true", default=False,
      help="List all available codecs and exits")
  
  supported_formats = sorted(FORMATS.keys())
  available_formats = sorted(ALL_FORMATS.keys())
  parser.add_argument("-f", "--format", metavar='FORMAT', type=str, nargs='*',
      default=supported_formats, choices=available_formats, help="The name of the format you want to test with. For a list of available formats, look below. If none given, run through all.")
  parser.add_argument("--list-formats", action="store_true", default=False,
      help="List all supported formats and exits")
  parser.add_argument("--list-all-formats", action="store_true", default=False,
      help="List all available formats and exits")

  parser.add_argument("-t", "--height", metavar="INT", type=int, 
      default=128, help="Height of the test video (defaults to %(default)s pixels). Note this number has to be even.")
  
  parser.add_argument("-w", "--width", metavar="INT", type=int, 
      default=128, help="Width of the test video (defaults to %(default)s pixels). Note this number has to be even.")

  parser.add_argument("-l", "--length", metavar="INT", type=int, 
      default=30, help="Length of the test sequence (defaults to %(default)s frames). The longer, the more accurate the test becomes.")

  parser.add_argument("-r", "--framerate", metavar="FLOAT", type=float,
      default=30., help="Framerate to be used on the test videos (defaults to %(default)s Hz).")

  parser.add_argument("-o", "--output", type=str,
      help="If set, then videos created for the tests are stored on the given directory. By default this option is empty and videos are created on a temporary directory and deleted after tests are done. If you set it, we also produced detailed output analysis for manual inspection.")

  parser.add_argument("-u", "--user-video", type=str, metavar="PATH",
      help="Set the path to the user video that will be used for distortion tests (if not set use default test video)")

  parser.add_argument("-n", "--user-frames", type=int, default=10, metavar="INT", help="Set the number of maximum frames to read from the user video (reads %(default)s by default)")

  args = parser.parse_args(args=user_input)

  # manual check because of argparse limitation
  for t in args.test:
    if t not in test_choices:
      parser.error("invalid test choice: '%s' (choose from %s)" % \
          (t, ", ".join(["'%s'" % k for k in test_choices])))

  if not args.test: args.test = test_choices

  if args.list_codecs: 
    print list_codecs()
    sys.exit(0)

  if args.list_all_codecs: 
    print list_all_codecs()
    sys.exit(0)

  if args.list_formats: 
    print list_formats()
    sys.exit(0)

  if args.list_all_formats:
    print list_all_formats()
    sys.exit(0)

  if 'user' in args.test and args.user_video is None:
    # in this case, take our standard video test
    args.user_video = test_utils.datafile("test.mov", io_test)

  def wrap_user_function(shape, framerate, format, codec, filename):
    return user_test(args.user_video, args.user_frames, format, codec, filename)

  # mapping between test name and function
  test_function = {
      'color': (utils.color_distortion, 'C'),
      'frameskip': (utils.frameskip_detection, 'S'),
      'noise': (utils.quality_degradation, 'N'),
      'user': (wrap_user_function, 'U'),
      }

  # result table
  table = {}

  print version_info
  print("Legend:")
  for k, (f, code) in test_function.iteritems():
    print("  %s: %s test" % (code, k.capitalize()))

  sys.stdout.write("Running %d test(s)..." %
      (len(args.test)*len(args.format)*len(args.codec)))
  sys.stdout.flush()

  # run tests
  for test in args.test:
    test_table = table.setdefault(test, {})
    f, code = test_function[test]

    for format in args.format:
      format_table = test_table.setdefault(format, {})

      for codec in args.codec:

        if codec not in supported_videowriter_formats()[format]['supported_codecs'].keys():
          # skip this test...
          sys.stdout.write(code)
          sys.stdout.flush()
          format_table[codec] = "%s+%s is unsupported" % (format, codec)
          continue

        if args.output:
  
          size = '%dx%dx%d@%gHz' % (args.length, args.height, args.width,
              args.framerate)
          outdir = os.path.join(args.output, test, codec, size, format)
          create_directories_save(outdir)

          try:
            result = detail(f, (args.length, args.height, args.width), 
                args.framerate, format, codec, outdir)
            sys.stdout.write(code)
            sys.stdout.flush()
          except Exception, e:
            result = str(e)
            sys.stdout.write(code)
            sys.stdout.flush()
          finally:
            format_table[codec] = result

        else:

          try:
            result = summarize(f, (args.length, args.height, args.width), 
                args.framerate, format, codec)
            sys.stdout.write(code)
            sys.stdout.flush()
          except Exception, e:
            result = str(e)
            sys.stdout.write(code)
            sys.stdout.flush()
          finally:
            format_table[codec] = result

  sys.stdout.write("\n")
  sys.stdout.flush()

  # report results in a readable way
  print version_info
  print "Settings:"
  print "  Width    : %d pixels" % args.width
  print "  Height   : %d pixels" % args.height
  print "  Length   : %d frames" % args.length
  print "  Framerate: %f Hz"     % args.framerate
  print ""

  print " %-9s | %-3s | %-16s | %s" % ('test', 'fmt', 'codec', 'figure (lower is better quality)')
  print ((11*'-') + '+' + (5*'-') + '+' + (18*'-') + '+' + ((80-45)*'-'))
  for test in sorted(table.iterkeys()):
    test_table = table[test]
    for format in sorted(test_table.iterkeys()):
      format_table = test_table[format]
      for codec in sorted(format_table.iterkeys()):
        figure = format_table[codec]
        print " %-9s | %-3s | %-16s | %s" % (test, format, codec, figure)
