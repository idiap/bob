#/usr/bin/env python

from bob.daq import *

def main():
  cam = V4LCamera("/dev/video0")
  pfs = cam.getSupportedPixelFormats()
  pf = pfs[0]
  fss = cam.getSupportedFrameSizes(pf)
  fs = fss[5]
  fis = cam.getSupportedFrameIntervals(pf, fs)
  fi = fis[1]


  cam.pixelFormat = pf
  cam.frameSize = fs
  cam.frameInterval = fi

  print pf
  print fs
  print fi

  a = CaptureSystem(cam)

  a.length = 15
  a.delay = 10
  a.name = 'myvideo'
  a.outputdir = '.'

  a.thumbnail = ''
  a.setExecuteOnStartRecording('echo "Start" &')
  a.setExecuteOnStopRecording('echo "Stop" &')
  a.setText("Custom text")

  a.fullscreen = True
  #a.setDisplaySize(200, 100)

  a.start()

if __name__ == '__main__':
  main()
