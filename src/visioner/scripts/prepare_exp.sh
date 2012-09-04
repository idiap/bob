#!/bin/bash

source common.sh

dir_src=`pwd`/../
dir_crt=`pwd`

# Build the project
mkdir -p ${dir_src}/build
cd ${dir_src}/build
cmake --verbose=0 ../ 
#make clean
make --quiet -j 3

# Copy the programs to the experimentation directory
cp ${dir_src}/build/projects/trainer ${dir_bin}/
cp ${dir_src}/build/projects/detector_eval ${dir_bin}/
cp ${dir_src}/build/projects/localizer_eval ${dir_bin}/
cp ${dir_src}/build/projects/classifier_eval ${dir_bin}/
cp ${dir_src}/build/projects/max_threads ${dir_bin}/

# Copy the scripts to the experimentation directory
cp ${dir_src}/scripts/common.sh ${dir_exp}
cp ${dir_src}/scripts/plot.sh ${dir_bin}/
cp ${dir_src}/scripts/task_face_detection.sh ${dir_exp}/
cp ${dir_src}/scripts/task_face_localization.sh ${dir_exp}/
cp ${dir_src}/scripts/task_face_pose_estimation.sh ${dir_exp}/

# Copy the baselines to the experimentation directory
cp ${dir_src}/scripts/baselines/baseline_* ${dir_baselines}

cd ${dir_crt}
