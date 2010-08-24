#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 02 Aug 2010 11:31:31 CEST 

"""Detects shaking behavior on a video stream
usage: shaking.py <video-file> [<scores> <number-of-seconds>]"""

import os, sys, math
import torch

def mean(v):
  return sum(v)/len(v)

def stdvar(v):
  m = mean(v)
  t = [(k-m)**2 for k in v]
  return (sum(t)/(len(t)-1))**0.5

def freq_analysis(v):
  # the nearest power-2 sized 1-d tensor
  power_of_2 = int(math.log(len(v))/math.log(2))
  t = torch.core.FloatTensor(2**power_of_2)
  for i in range(t.size(0)): t.set(i, v[i])
  fft = torch.sp.spFFT()
  if not fft.process(t):
    raise RuntimeError, 'spFFT did not process correctly!'
  out = fft.getOutput(0)
  abs_values = [] 
  for k in range(out.size(0)):
    abs_values.append(((out.get(k, 0)**2) + (out.get(k, 1)**2))**0.5)
  return sum(abs_values[1:])/sum(abs_values[:1])

def main():
  if len(sys.argv) < 2 or len(sys.argv) > 4:
    print __doc__ 
    sys.exit(1)

  input = torch.ip.Video(sys.argv[1])
  buffer = torch.ip.Image(1, 1, 1) #forces gray-scale conversion
  previous = torch.ip.Image(1, 1, 1) #forces gray-scale conversion

  scores = None
  if len(sys.argv) > 2:
    score_dir = os.path.dirname(sys.argv[2])
    if not os.path.exists(score_dir): os.makedirs(score_dir)
    scores = file(sys.argv[2], 'wt')

  max_frames = input.nframes
  if len(sys.argv) > 3:
    time = int(sys.argv[3])
    max_frames = time * input.framerate
    max_frames = min(max_frames, input.nframes)

  # start the work here...
  input.read(previous) #loads frame #0

  frame = 1
  values = []
  while (input.read(buffer) and (frame < max_frames)):
    save = torch.ip.Image(buffer)
    buffer -= previous
    previous = save
    buffer.reset(0, 0)
    values.append(buffer.sum()/buffer.sizeAll())
    if scores: scores.write('%.3e\n' % values[-1])
    frame += 1

  # calculates the ratio between FFT magnitudes at 0 and the sum of the rest.
  fa = freq_analysis(values)

  if os.path.basename(sys.argv[1]).find('attack') == 0: print "attack",
  else: print "real",
  print '%.3e %.3e %.3e %.3e %.3e %s' % (mean(values), stdvar(values), min(values), max(values), fa, os.path.basename(os.path.realpath(sys.argv[1])))
  
  input.close()
  if scores: scores.close()

if __name__ == '__main__':
  main()

