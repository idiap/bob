#!/bin/sh

image="/idiap/common_vision/visidiap/databases/cmu-test/images_gray/addams-family.pgm"
#model="../data/models/facedetection/frontal/mct4.cascade"
#model="/idiap/bvenka/cosmin/modelHaar23jun09-1.wsm"
#model="/idiap/bvenka/cosmin/LBP-Cascade-model32.wsm"
model="/idiap/bvenka/cosmin/modelHaarCascadeOpenCV-32.wsm"
#model="../data/models/boosting/modelHaarCascade.wsm"
	#modelHaarCascade.wsm
	#modelHaarSingleStage.wsm
	#modelHLBPCascade_3_stages_full.wsm
	#modelHLBPCascade.wsm
	#modelHLBPSingleStage.wsm
	#modelLBPCascade.wsm
	#model-lbp-single-stage.wsm

# Scanning parameters:
#	LBP: multiscale (-prep_ii) vs. pyramid ()
#	Haar: multiscale (-prep_ii) vs. pyramid (-prep_ii)
#	HLBP: multiscale (-prep_hlbp -prep_ii) vs. pyramid (-prep_hlbp -prep_ii)
params_prune=""	#-prune_use_mean -prune_use_stdev"
params_select="-select_type 1 -select_merge_type 1 -select_min_surf_overlap 60" #-select_overlap_iterative
params_general="-dx 0.1 -dy 0.1 -ds 1.5 -min_patt_w 0 -max_patt_w 10000 -min_patt_h 0 -max_patt_h 10000"

# Multiscale
params=$params_prune" "$params_select" "$params_general" -verbose -prep_ii" #-prep_hlbp -prep_ii"
./`uname -s`_`uname -m`/scanning $image $model -explorer_type 1 -scale_explorer_type 0 $params

# Pyramid
params=$params_prune" "$params_select" "$params_general" -verbose -prep_ii" #-prep_hlbp -prep_ii"
./`uname -s`_`uname -m`/scanning $image $model -explorer_type 0 -scale_explorer_type 0 $params

#gdb --args
#valgrind --leak-check=full
