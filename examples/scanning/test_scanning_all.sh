#!/bin/sh

image_dir="/idiap/common_vision/visidiap/databases/cmu-test/images_gray"
face_model="../data/models/facedetection/frontal/mct4.cascade"
context_model="/idiap/catana/experiments/cmu_test/mct4.wsm/dx_0.1_dy_0.1_ds_1.5/context_models/face.context"

# Scanning parameters:
#	LBP: multiscale (-prep_ii) vs. pyramid ()
#	Haar: multiscale (-prep_ii) vs. pyramid (-prep_ii)
#	HLBP: multiscale (-prep_hlbp -prep_ii) vs. pyramid (-prep_hlbp -prep_ii)
#	-----------
#	explorer_type: 0 - Pyramid, 1 - MultiScale, 2 - Greedy
#	scale_explorer_type: 0 - Exhaustive, 1 - Spiral, 2 - Random, 3 - Mixed
#	select_type: 0 - Overlap, 1 - AMS
#	select_merge_type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence
#	-stop_at_first_detection, -start_with_large_scale
#	context_type: 0 - Full, 1 - Axis
params_prune=""	#-prune_use_mean -prune_use_stdev"
params_select="-select_type 1 -select_merge_type 1 -select_min_surf_overlap 60" #-select_overlap_iterative
params_general="-dx 0.1 -dy 0.1 -ds 1.5 -min_patt_w 0 -max_patt_w 10000 -min_patt_h 0 -max_patt_h 10000"
params_context="-context_model $context_model  -context_type 1"

# Scan each image in the directory
for image in `ls $image_dir/*.pgm`
do
	# Multiscale
	params=$params_prune" "$params_select" "$params_general" "$params_context" -verbose -prep_ii"
	./`uname -s`_`uname -m`/scanning $image $face_model -explorer_type 1 -scale_explorer_type 0 $params

	# Pyramid
	#params=$params_prune" "$params_select" "$params_general" "$params_context" -verbose"
	#./`uname -s`_`uname -m`/scanning $image $face_model -explorer_type 0 -scale_explorer_type 0 $params

	# Greedy
	params=$params_prune" "$params_select" "$params_general" "$params_context" -verbose -prep_ii"
	./`uname -s`_`uname -m`/scanning $image $face_model -explorer_type 2 -scale_explorer_type 0 $params
done

