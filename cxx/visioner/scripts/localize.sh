#!/bin/bash

# Localization parameters
data=/home/cosmin/idiap/databases/faces/bioid/all.list
results=./	# Directory to save images

detect_model=/home/cosmin/idiap/experiments/face_detection/models/fface20x24.mct8.ept.model
#detect_model=models/fface20x24.model
detect_thres=10.0	# Model threshold
detect_levels=4         # Greedy scanning #levels (0 - slow and safe, >0 - faster but can miss objects)
detect_ds=2		# Scanning resolution (only for classification)
detect_cluster=0.01	# Non-maxima suppression clustering threshold of the thresholded detections
                        #	if 0.00 then only the detection with the maximum score will be selected
detect_method=scanning  # Detection mode: groundtruth, scanning  
#detect_method=groundtruth # Detection mode: groundtruth, scanning                        

localize_model=/home/cosmin/idiap/experiments/face_localization_leye_reye/models/ffacial80x96.mct8.ept.model
localize_method=mshots+med   # localization method: 1shot, mshots+avg, mshots+med

# Run the localizer
params=""
params=${params}"--data ${data} "
params=${params}"--results ${results} "
params=${params}"--detect_model ${detect_model} "
params=${params}"--detect_threshold ${detect_thres} "
params=${params}"--detect_levels ${detect_levels} "
params=${params}"--detect_ds ${detect_ds} "
params=${params}"--detect_cluster ${detect_cluster} "
params=${params}"--detect_method ${detect_method} "
params=${params}"--localize_model ${localize_model} "
params=${params}"--localize_method ${localize_method} "

./build/projects/localizer ${params}
