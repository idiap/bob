#!/bin/sh

image="../data/images/1001_f_g1_s01_1001_en_1.jpeg"

face_model="../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade"
context_model="/idiap/catana/experiments/cmu_cmu/dx_0.1_dy_0.1_ds_1.1/axis/mct4.wsmxxx-0.049904xxx/context_models/face.context"
#context_model="/idiap/catana/experiments/zzz_old/zzz_old_context/cmu_test/mct4.wsm/dx_0.1_dy_0.1_ds_1.5/context_models/face.context"

#face_model="/idiap/bvenka/cosmin/modelHaar23jun09-1.wsm"
#face_model="/idiap/bvenka/cosmin/LBP-Cascade-model32.wsm"
#face_model="/idiap/bvenka/Ani/modelHLBP_29jun_t3_cascade.wsm"
#face_model="/idiap/bvenka/Ani/modelHLBP_29jun_t4_cascade.wsm"
#face_model="/idiap/bvenka/cosmin/modelHaarCascadeOpenCV-32.wsm"
#face_model="../data/models/boosting/modelHaarCascade.wsm"

# Scanning parameters:
#	LBP: multiscale (-prep_ii) vs. pyramid ()
#	Haar: multiscale (-prep_ii) vs. pyramid (-prep_ii)
#	HLBP: multiscale (-prep_hlbp -prep_ii) vs. pyramid (-prep_hlbp -prep_ii)
#	-----------
#	explorer_type: 0 - Pyramid, 1 - MultiScale, 2 - Greedy
#	scale_explorer_type: 0 - Exhaustive, 1 - Spiral, 2 - Random
#	select_type: 0 - Overlap, 1 - AMS
#	select_merge_type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence
#	-stop_at_first_detection, -start_with_large_scale
#	context_type: 0 - Full, 1 - Axis
params_prune=" " #-prune_use_mean -prune_use_stdev -prune_min_mean 25 -prune_max_mean 225 -prune_min_stdev 20 -prune_max_stdev 125"
params_general="-dx 0.1 -dy 0.1 -ds 1.1 -min_patt_w 0 -max_patt_w 10000 -min_patt_h 0 -max_patt_h 10000"
params_context="-context_model $context_model -context_type 1"

# Multiscale
params_select="-select_type 0 -select_merge_type 1 -select_min_surf_overlap 60" #-select_overlap_iterative
params=$params_prune" "$params_select" "$params_general" "$params_context" -prep_ii" #-verbose"
time ./`uname -s`_`uname -m`/scanning $image $face_model -explorer_type 1 -scale_explorer_type 0 $params

# Pyramid
params_select="-select_type 0 -select_merge_type 1 -select_min_surf_overlap 60" #-select_overlap_iterative
params=$params_prune" "$params_select" "$params_general" "$params_context" " # -verbose"
time ./`uname -s`_`uname -m`/scanning $image $face_model -explorer_type 0 -scale_explorer_type 0 $params

## Context
#params_select="-select_type 1 -select_merge_type 1 -select_min_surf_overlap 60" #-select_overlap_iterative
#params=$params_prune" "$params_select" "$params_general" "$params_context" -verbose -prep_ii"
#time ./`uname -s`_`uname -m`/scanning $image $face_model -explorer_type 2 -scale_explorer_type 0 $params

#gdb --args
#valgrind --leak-check=full
