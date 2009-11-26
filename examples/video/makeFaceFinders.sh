#!/bin/sh

face_model="../data/models/facedetection/frontal/mct4.cascade"
context_model="../data/models/facedetection/frontal/mct4.banca.dx01dy01ds11.defthres.full.context"

# Scanning parameters:
#	LBP: multiscale (-prep_ii) vs. pyramid ()
#	Haar: multiscale (-prep_ii) vs. pyramid (-prep_ii)
#	HLBP: multiscale (-prep_hlbp -prep_ii) vs. pyramid (-prep_hlbp -prep_ii)
#	-----------
#	explorer_type: 0 - Pyramid, 1 - MultiScale, 2 - Greedy
#	scale_explorer_type: 0 - Exhaustive, 1 - Spiral, 2 - Random
#	select_type: 0 - Overlap, 1 - AMS, 2 - No merge
#	select_overlap_type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence
#	-stop_at_first_detection, -start_with_large_scale
#	context_type: 0 - Full, 1 - Axis
params_prune=" " #-prune_use_mean -prune_use_stdev -prune_min_mean 25 -prune_max_mean 225 -prune_min_stdev 20 -prune_max_stdev 125"
params_general="-dx 0.1 -dy 0.1 -ds 1.1 -min_patt_w 0 -max_patt_w 10000 -min_patt_h 0 -max_patt_h 10000"
params_context="-context_model $context_model -context_type 0"

# Multiscale
params_select="-select_type 0 -select_overlap_type 1 -select_overlap_min_surf 60" #-select_overlap_iterative
params=$params_prune" "$params_select" "$params_general" "$params_context" -prep_ii" #-verbose"
./`uname -s`_`uname -m`/makeFaceFinder "facefinder.multiscale.params" $face_model -explorer_type 1 -scale_explorer_type 0 $params

# Pyramid
params_select="-select_type 0 -select_overlap_type 1 -select_overlap_min_surf 60" #-select_overlap_iterative
params=$params_prune" "$params_select" "$params_general" "$params_context" " # -verbose"
./`uname -s`_`uname -m`/makeFaceFinder "facefinder.pyramid.params" $face_model -explorer_type 0 -scale_explorer_type 0 $params

## Context
params_select="-select_type 1"
params=$params_prune" "$params_select" "$params_general" "$params_context" -prep_ii" #-verbose"
./`uname -s`_`uname -m`/makeFaceFinder "facefinder.context.params" $face_model -explorer_type 2 -scale_explorer_type 0 $params

