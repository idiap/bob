#!/bin/bash

source common.sh

#######################################################################
# Directories
#######################################################################

dir_task=${dir_exp}/cifar10_recognition/

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
data_train=${data_train}${dir_db}/cifar10/train.batch1.list:
data_train=${data_train}${dir_db}/cifar10/train.batch2.list:
data_train=${data_train}${dir_db}/cifar10/train.batch3.list:
data_train=${data_train}${dir_db}/cifar10/train.batch4.list:
data_train=${data_train}${dir_db}/cifar10/train.batch5.list:

data_valid=""
data_valid=${data_valid}${dir_db}/cifar10/valid.batch1.list:
data_valid=${data_valid}${dir_db}/cifar10/valid.batch2.list:
data_valid=${data_valid}${dir_db}/cifar10/valid.batch3.list:
data_valid=${data_valid}${dir_db}/cifar10/valid.batch4.list:
data_valid=${data_valid}${dir_db}/cifar10/valid.batch5.list:

data_test=""
data_test=${data_valid}${dir_db}/cifar10/test.list:

test_name="CIFAR10"

#######################################################################
# Model parameters
#######################################################################

labels="airplane:automobile:bird:cat:deer:dog:frog:horse:ship:truck"

common_params=""
common_params=${common_params}"--model_labels=${labels},"	# Labels
common_params=${common_params}"--model_ds=2,"                   # Scanning resolution to collect training samples
common_params=${common_params}"--model_rows=32,"                # Model size
common_params=${common_params}"--model_cols=32,"
common_params=${common_params}"--model_min_gt_overlap=0.90,"	# Minimum overlapping with GT
common_params=${common_params}"--model_tagger=object_type,"	# Tagger: object types
common_params=${common_params}"--model_train_data=${data_train},"   # Training data
common_params=${common_params}"--model_valid_data=${data_valid},"   # Validation data
common_params=${common_params}"--model_rounds=1024,"		# #rounds
common_params=${common_params}"--model_shrinkage=1.0,"          # Shrinkage
common_params=${common_params}"--model_loss=log,"               # Loss
common_params=${common_params}"--model_loss_param=0.0,"         # Loss parameter

#######################################################################
# Models
#######################################################################

#model_names=(	"MCT4"                 
#                "MCT6" 
#                "MCT8" 
#                "MCT9"
#                "MCT10"
#                "MCT12")

#model_params=(	"${nonoverlap},${mct4},${gboost_shared_ept},"                
#                "${nonoverlap},${mct6},${gboost_shared_ept},"
#                "${nonoverlap},${mct8},${gboost_shared_ept},"
#                "${nonoverlap},${mct9},${gboost_shared_ept},"
#                "${nonoverlap},${mct10},${gboost_shared_ept},"
#                "${nonoverlap},${mct12},${gboost_shared_ept},")
                
model_names=(	"MCT6.SHARED.EPT"                 
                "MCT6.SHARED.VAR"
                "MCT6.INDEP.EPT"                                 
                "MCT6.INDEP.VAR")

model_params=(	"${nonoverlap},${mct6},${gboost_shared_ept},"
                "${nonoverlap},${mct6},${gboost_shared_var},"
                "${nonoverlap},${mct6},${gboost_indep_ept},"                
                "${nonoverlap},${mct6},${gboost_indep_var},")

detect_names=()
detect_params=()
for (( k=0; k<${#model_names[*]}; k++ ))
do
        detect_names=("${detect_names[@]}" "CIFAR10.${model_names[$k]}")
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
        jparams=${jparams}" --detect_levels 0"
        jparams=${jparams}" --detect_ds 2"
        jparams=${jparams}" --detect_cluster 0.01"

        log=${dir_logs}/Test.${jname}.${test_name}.log
        job=${dir_task}/job_test.${jname}.${test_name}.sh

        echo "#!/bin/bash" > ${job}
        echo "${dir_bin}/detector_eval ${jparams} > ${log}" >> ${job}
        chmod +x ${job}
done

#######################################################################
# Train and test the models
#######################################################################

run_training_scripts_local
run_testing_scripts_local detector_eval

