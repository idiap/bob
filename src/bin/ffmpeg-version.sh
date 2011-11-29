#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 29 Nov 2011 13:36:43 CET

string=`ffmpeg -version 2>&1 | grep -i 'ffmpeg version'`;
python -c "print '${string}'.split(' ')[2].strip(',')"
