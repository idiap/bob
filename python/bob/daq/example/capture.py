#!/usr/bin/env python

from bob.daq import *

def main():
  cam = V4LCamera("/dev/video0")
  pfs = cam.get_supported_pixel_formats()
  pf = pfs[0]
  fss = cam.get_supported_frame_sizes(pf)
  fs = fss[5]
  fis = cam.get_supported_frame_intervals(pf, fs)
  fi = fis[1]


  cam.pixel_format = pf
  cam.frame_size = fs
  cam.frame_interval = fi

  print pf
  print fs
  print fi

  a = CaptureSystem(cam)

  a.length = 15
  a.delay = 10
  a.name = 'myvideo'
  a.outputdir = '.'

  a.thumbnail = ''
  a.set_execute_on_start_recording('echo "Start" &')
  a.set_execute_on_stop_recording('echo "Stop" &')
  a.set_text("Custom text")

  a.fullscreen = True
  #a.set_display_size(200, 100)

  a.start()

if __name__ == '__main__':
  main()
