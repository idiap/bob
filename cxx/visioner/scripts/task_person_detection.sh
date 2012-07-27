#!/bin/bash

source common.sh

#######################################################################
# Directories
#######################################################################

dir_task=${dir_exp}/person_detection/

dir_models=${dir_task}/models/
mkdir -p ${dir_models}

dir_results=${dir_task}/results/
mkdir -p ${dir_results}

dir_plots=${dir_task}/plots/
mkdir -p ${dir_plots}

dir_logs=${dir_task}/logs/
mkdir -p ${dir_logs}

#######################################################################
# Training and testing datasets
#######################################################################

data_train=""
data_train=${data_train}${dir_db}/inria-person/train.neg.list:
data_train=${data_train}${dir_db}/inria-person/train.pos.list:

data_valid=""
data_valid=${data_valid}${dir_db}/inria-person/valid.neg.list:
data_valid=${data_valid}${dir_db}/inria-person/valid.pos.list:

data_tests=(	${dir_db}/inria-person/test.neg.list:${dir_db}/inria-person/test.pos.list:
                )

test_names=(	INRIA
                )

#######################################################################
# Model parameters
#######################################################################

common_params=""
common_params=${common_params}"--model_labels=person,"	# Person type
common_params=${common_params}"--model_seed=0,"		# Random seed: for sampling training samples
common_params=${common_params}"--model_ds=2,"		# Scanning resolution to collect training samples
common_params=${common_params}"--model_rows=64,"	# Model size
common_params=${common_params}"--model_cols=32,"
common_params=${common_params}"--model_min_gt_overlap=0.80,"	# Minimum overlapping with GT
common_params=${common_params}"--model_tagger=object_type,"	# Tagger: object types
common_params=${common_params}"--model_train_data=${data_train},"       # Training data
common_params=${common_params}"--model_valid_data=${data_valid},"       # Validation data
common_params=${common_params}"--model_valid_samples=5000,"             # #validation samples
common_params=${common_params}"--model_train_samples=10000,"            # #training samples
common_params=${common_params}"--model_rounds=1024,"		# #rounds

TODO
exit

#######################################################################
# Models
#######################################################################

model_names=(	"MCT2" 
                "MCT3" 
                "MCT4"                 
                "MCT6" 
                "MCT8" 
                "MCT9")

model_params=(	"${mct2},${log},${gboost_indep_ept},"                
                "${mct3},${log},${gboost_indep_ept},"
                "${mct4},${log},${gboost_indep_ept},"
                "${mct6},${log},${gboost_indep_ept},"
                "${mct8},${log},${gboost_indep_ept},"
                "${mct9},${log},${gboost_indep_ept},")

model_styles=""
model_styles=${model_styles}"lt:1:lc:rgb:\"red\":pt:1:lw:3,"
model_styles=${model_styles}"lt:1:lc:rgb:\"pink\":pt:1:lw:3,"
model_styles=${model_styles}"lt:1:lc:rgb:\"orange\":pt:1:lw:3,"
model_styles=${model_styles}"lt:1:lc:rgb:\"green\":pt:1:lw:3,"
model_styles=${model_styles}"lt:1:lc:rgb:\"blue\":pt:1:lw:3,"
model_styles=${model_styles}"lt:1:lc:rgb:\"black\":pt:1:lw:3,"

detect_names=()
detect_params=()
for (( k=0; k<${#model_names[*]}; k++ ))
do
        detect_names=("${detect_names[@]}" "Person.${model_names[$k]}")
        detect_params=("${detect_params[@]}" "${common_params},${model_params[$k]}")
done

#######################################################################
# Generate the training scripts
#######################################################################

echo "<<< Preparing the scripts to train the models ..."

rm -f ${dir_task}/job_train*

for (( i=0; i<${#detect_names[*]}; i++ ))
do
        jname=${detect_names[$i]}
        jparams=${detect_params[$i]}
        jparams=${jparams//,/ }

        log=${dir_logs}/${jname}.train.log
        job=${dir_task}/job_train.${jname}.sh
        rm -f ${job}

        jparams=${jparams}" --model ${dir_models}/${jname}"
        echo "#!/bin/bash" >> ${job}
        echo "${dir_bin}/trainer ${jparams} > ${log}" >> ${job}

        chmod +x ${job}
done

#######################################################################
# Generate the detection scripts
#######################################################################

echo "<<< Preparing the detection scripts ..."

rm -f ${dir_task}/job_test*

# Testing dataset ...
for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
do
        data_test=${data_tests[$i_data]}
        test_name=${test_names[$i_data]}

        dir_result=${dir_results}/${test_name}
        mkdir -p ${dir_result}

        # Models
        for (( i=0; i<${#detect_names[*]}; i++ ))
        do
                jname=${detect_names[$i]}

                jparams=""
                jparams=${jparams}" --data ${data_test}"
                jparams=${jparams}" --roc ${dir_result}/${jname}"
                jparams=${jparams}" --detect_model ${dir_models}/${jname}"
                jparams=${jparams}" --detect_threshold 0.0"
                jparams=${jparams}" --detect_levels 6"
                jparams=${jparams}" --detect_ds 2"
                jparams=${jparams}" --detect_cluster 0.01"

                log=${dir_logs}/Test.${jname}.${test_name}.log
                job=${dir_task}/job_test.${jname}.${test_name}.sh

                echo "#!/bin/bash" > ${job}
                echo "${dir_bin}/detector_eval ${jparams} > ${log}" >> ${job}
                chmod +x ${job}
        done
done

#######################################################################
# Train and test the models
#######################################################################

run_training_scripts_local
run_testing_scripts_local detector_eval

#######################################################################
# Plot the ROC curves
#######################################################################

echo ">>> Plotting the ROC curves ..."

# Test dataset ...
for (( i_data=0; i_data < ${#data_tests[*]}; i_data++ ))
do
        test_name=${test_names[$i_data]}
        echo "<<< ${test_name} >>> ..."

        base_input=${dir_results}/${test_name}/

        base_dir=${dir_plots}/${test_name}
        mkdir -p ${base_dir}

        # Vary the models ...
        plot_labels="FA,DR"
        plot_data="2:1"
        plot_limits="0:10000,0.00:0.50"
        plot_styles=${model_styles}
        plot_title=${test_name}
        plot_eps=${base_dir}/detection.${test_name}.eps

        plot_inputs=""
        plot_names=""
        for (( i=0; i<${#detect_names[*]}; i++ ))
        do
                plot_names=${plot_names},${detect_names[$i]}
                plot_inputs=${plot_inputs}${base_input}/${detect_names[$i]}.roc,
        done

        plot_params=${plot_inputs}" "${plot_names}" "${plot_styles}" "${plot_title}
        plot_params=${plot_params}" "${plot_labels}" "${plot_data}" "${plot_limits}" "${plot_eps}
        bash ${dir_bin}/plot.sh ${plot_params}
done
