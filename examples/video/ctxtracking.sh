#!/bin/bash

facefinder="facefinder.track.context.params"
dir_videos="./videos"
dir_results="./results"

for video in `ls ${dir_videos}/*`
do
      result=${dir_results}/ctxtrack_`basename ${video}`
      
      echo "Processing video [${video}], output video [${result}] ..."	
      ./`uname -s`_`uname -m`/fctxtrackingVideo ${video} ${facefinder} ${result}
      echo "---------------------------------------------------------"
done
