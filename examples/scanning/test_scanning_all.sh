#!/bin/sh

image_dir="../data/cmu_faces"
model="../data/models/facedetection/frontal/mct4.cascade"

params_select="-select_type 0 -select_merge_type 1 -select_overlap_iterative -select_min_surf_overlap 60"
params_prune=""	#-prune_use_mean -prune_use_stdev"
params_general="-dx 0.0 -dy 0.0 -ds 1.0 -min_patt_w 0 -max_patt_w 10000 -min_patt_h 0 -max_patt_h 10000"
params=$params_prune" "$params_select" "$params_general" -verbose"

# Scan each image in the directory
for image in `ls $image_dir/*.pgm`
do
	# Pyramid
	./Linux_i686/scanning $image $model -explorer_type 0 -scale_explorer_type 0 $params

	# Multiscale
	./Linux_i686/scanning $image $model -explorer_type 1 -scale_explorer_type 0 $params
done

