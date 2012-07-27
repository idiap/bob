#!/bin/bash

# Detection parameters
data=/home/cosmin/idiap/databases/faces/mit+cmu/all.list
results=./	# Directory to save images / bounding boxes

detect_model=/home/cosmin/idiap/experiments/face_detection/models/face20x24.elbp.ept.model
detect_thres=23.0	# Model threshold
detect_levels=10         # Greedy scanning #levels (0 - slow and safe, >0 - faster but can miss objects)
detect_ds=2		# Scanning resolution (only for classification)
detect_cluster=0.05	# Non-maxima suppression clustering threshold of the thresholded detections
                        #	if 0.00 then only the detection with the maximum score will be selected
detect_method=scanning  # Detection mode: groundtruth, scanning

# Run the detector
params=""
params=${params}"--data ${data} "
params=${params}"--results ${results} "
params=${params}"--detect_model ${detect_model} "
params=${params}"--detect_threshold ${detect_thres} "
params=${params}"--detect_levels ${detect_levels} "
params=${params}"--detect_ds ${detect_ds} "
params=${params}"--detect_cluster ${detect_cluster} "
params=${params}"--detect_method ${detect_method} "

./build/projects/detector ${params}
#./build/projects/detector2bbx ${params}
