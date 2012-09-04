#!/bin/bash

param2model=../../build/projects/param2model
loc_eval_ex=../../build/projects/localizer_eval_ex

dir_asm=/home/cosmin/idiap/databases/faces/ASM-AAM-Idiap/experiments/
dir_db=/home/cosmin/idiap/databases/

ff_configs="leye:reye leye:reye:ntip:lmc:rmc leye:reye:ntip:chin leye:reye:lmc:rmc:ntip:chin leye:reye:leic:leoc:reic:reoc:lmc:rmc:ntip:tlip:blip:chin"

data_test=""
data_test=${data_test}${dir_db}/faces/multipie/test.04_1.list:
data_test=${data_test}${dir_db}/faces/multipie/test.05_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.05_1.list:
data_test=${data_test}${dir_db}/faces/multipie/test.13_0.list:
data_test=${data_test}${dir_db}/faces/multipie/test.14_0.list:

data_tests=(	${data_test}
		${dir_db}/faces/bioid/all.list
		${dir_db}/faces/xm2vts/all.list)

test_names=(	MULTIPIE
		BIOID
		XM2VTS)

for ffs in ${ff_configs}
do
	model=${ffs//:/_}.model

	${param2model} --model_rows 48 --model_cols 40 --model_labels $ffs --model_tagger keypoint --model ${model}

	for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
        do
                data=${data_tests[$i_data]}
                name=${test_names[$i_data]}		
		pred=${dir_asm}/${name}.list
		loc=baseline_localization_${name}_${ffs//:/_}		

		${loc_eval_ex} --data ${data} --predictions ${pred} --loc ${loc} --localize_model ${model}
	done

	rm *.histo.*
	rm *.model
done
